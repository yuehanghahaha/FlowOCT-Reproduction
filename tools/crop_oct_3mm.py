"""裁剪 OCTA-3mm 的 OCT/OCTA BMP，另存且保留原像素。

每个病例的全部切片共用同一个上下范围；只裁上下，不缩放、不补边。
默认根据 GT_Layers 裁剪 OCT；OCTA 直接复用已裁剪 OCT 的坐标表。
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from scipy.io import loadmat


def crop_bounds(layer_path, scan_count, width, height, margin_top, margin_bottom):
    layer = loadmat(layer_path)['Layer'].astype(np.float64)
    if layer.shape != (6, scan_count, width):
        raise ValueError(f'Layer 形状不匹配: {layer_path}: {layer.shape}')
    if not np.isfinite(layer).all() or (layer < 1).any() or (layer > height).any():
        raise ValueError(f'层标注含无效坐标: {layer_path}')
    if (np.diff(layer, axis=0) < 0).any():
        raise ValueError(f'层标注上下顺序反转: {layer_path}')
    # MAT 中存储的是 MATLAB 行号；减 1 后得到 Python 0 基坐标。
    top = float(layer[0].min()) - 1
    bottom = float(layer[-1].max()) - 1
    z0 = max(0, math.floor(top) - margin_top)
    z1 = min(height, math.ceil(bottom) + margin_bottom + 1)
    return z0, z1


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True,
                        help='包含 OCTA_3mm 和 Label 的 OCTA-500 根目录')
    parser.add_argument('--output', type=Path, required=True, help='独立的新输出目录')
    parser.add_argument('--modality', choices=['OCT', 'OCTA'], default='OCT')
    parser.add_argument('--coordinates', type=Path,
                        help='复用已裁剪 OCT 的 crop_coordinates.csv，OCTA 模式必须提供')
    parser.add_argument('--margin-top', type=int, default=30)
    parser.add_argument('--margin-bottom', type=int, default=120)
    parser.add_argument('--cases', nargs='+', help='只处理指定病例；省略则处理全部 200 例')
    args = parser.parse_args()
    if args.margin_top < 0 or args.margin_bottom < 0:
        parser.error('保留余量不能为负数')
    root = args.dataset_root.resolve()
    source = root / 'OCTA_3mm' / args.modality
    layers = root / 'Label' / 'GT_Layers'
    output = args.output.resolve()
    if not source.is_dir() or not layers.is_dir():
        parser.error(f'数据根目录内必须存在 OCTA_3mm/{args.modality} 和 Label/GT_Layers')
    if args.modality == 'OCTA' and args.coordinates is None:
        parser.error('OCTA 模式请提供已裁剪 OCT 的 --coordinates，确保配对坐标一致')
    reference = None
    coordinate_path = None
    if args.coordinates is not None:
        coordinate_path = args.coordinates.resolve()
        with coordinate_path.open(encoding='utf-8-sig', newline='') as file:
            reference_rows = list(csv.DictReader(file))
        reference = {row['case']: row for row in reference_rows}
        if len(reference) != len(reference_rows):
            parser.error('坐标表中的病例编号重复')
        with (coordinate_path.parent / 'crop_config.json').open(encoding='utf-8') as file:
            reference_config = json.load(file)
        # 余量以实际参考导出配置为准；裁剪范围不在 OCTA 上重新计算。
        args.margin_top = reference_config['margin_top']
        args.margin_bottom = reference_config['margin_bottom']
    # 输出不能位于原始数据子集或标签目录内，避免混淆和误覆盖。
    for original in (root / 'OCTA_3mm', root / 'Label'):
        if output == original or output.is_relative_to(original) or original.is_relative_to(output):
            parser.error('输出必须与原始 OCTA_3mm、Label 目录分开')
    if coordinate_path is not None:
        original = coordinate_path.parent
        if output == original or output.is_relative_to(original) or original.is_relative_to(output):
            parser.error('输出必须与参考 OCT 裁剪目录分开')
    if output.exists() and any(output.iterdir()):
        parser.error('输出目录已有内容，请指定新的空目录；本程序不覆盖已有数据')
    cases = sorted(p for p in source.iterdir() if p.is_dir())
    if args.cases:
        requested = set(args.cases)
        missing = requested - {p.name for p in cases}
        if missing:
            parser.error(f'病例目录不存在: {sorted(missing)}')
        cases = [p for p in cases if p.name in requested]
    if not cases:
        parser.error('没有可处理的病例')

    # 先检查全部病例的输入列表和标注，再创建输出。
    plans = []
    for folder in cases:
        files = sorted(folder.glob('*.bmp'), key=lambda p: int(p.stem))
        if [int(p.stem) for p in files] != list(range(1, 305)):
            raise ValueError(f'{folder.name}: 预期连续 304 张 BMP，实际 {len(files)} 张')
        with Image.open(files[0]) as image:
            if image.mode != 'L' or image.size != (304, 640):
                raise ValueError(f'{files[0]}: 预期 304×640 灰度图，实际 {image.size}/{image.mode}')
            width, height = image.size
        label_path = layers / (folder.name + '.mat')
        if reference is None:
            z0, z1 = crop_bounds(label_path, len(files), width, height,
                                 args.margin_top, args.margin_bottom)
        else:
            if folder.name not in reference:
                raise ValueError(f'{folder.name}: 参考坐标表缺少该病例')
            row = reference[folder.name]
            if row['complete'] != 'True' or int(row['saved_verified_slices']) != len(files):
                raise ValueError(f'{folder.name}: 参考 OCT 未完成或切片数不一致')
            if (int(row['original_width']), int(row['original_height'])) != (width, height):
                raise ValueError(f'{folder.name}: OCT/OCTA 原图尺寸不一致')
            z0, z1 = int(row['z0']), int(row['z1_exclusive'])
            if not (0 <= z0 < z1 <= height) or z1-z0 != int(row['cropped_height']):
                raise ValueError(f'{folder.name}: 参考裁剪坐标无效')
            reference_files = {p.name for p in (coordinate_path.parent / folder.name).glob('*.bmp')}
            if reference_files != {p.name for p in files}:
                raise ValueError(f'{folder.name}: 已裁剪 OCT 与原始 OCTA 的文件名不匹配')
        plans.append((folder, files, width, height, z0, z1, label_path))

    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'crop_config.json', {
        'dataset_root': str(root), f'source_{args.modality.lower()}': str(source),
        'output': str(output), 'modality': args.modality,
        'coordinate_source': 'existing OCT crop coordinates' if reference is not None else 'GT_Layers annotation',
        'reference_coordinates': str(coordinate_path) if coordinate_path is not None else None,
        'layer_order': ['ILM', 'IPL', 'OPL', 'ISOS', 'RPE', 'BM'],
        'source_layer_coordinates': 'MATLAB 1-based rows; converted to Python 0-based',
        'crop_interval': '[z0, z1_exclusive)',
        'margin_top': args.margin_top, 'margin_bottom': args.margin_bottom,
        'resize': False, 'padding': False, 'pixel_verification': 'every saved BMP',
        'case_count': len(plans), 'source_files_modified': False,
    })
    metadata = []
    failures = []
    saved_count = 0
    started = time.monotonic()
    for index, (folder, files, width, height, z0, z1, label_path) in enumerate(plans, 1):
        destination = output / folder.name
        destination.mkdir()
        case_saved = 0
        for path in files:
            saved_path = destination / path.name
            try:
                with Image.open(path) as original:
                    if original.mode != 'L' or original.size != (width, height):
                        raise ValueError(f'切片尺寸/模式异常: {original.size}/{original.mode}')
                    source_array = np.asarray(original)
                    expected = source_array[z0:z1, :]
                    cropped = original.crop((0, z0, width, z1))
                    cropped.save(saved_path, format='BMP')
                with Image.open(saved_path) as reread:
                    if reread.mode != 'L' or reread.size != (width, z1-z0):
                        raise ValueError('保存后尺寸/模式不一致')
                    if not np.array_equal(np.asarray(reread), expected):
                        raise ValueError('保存后的像素与原图对应区域不一致')
                case_saved += 1
                saved_count += 1
            except Exception as error:
                failures.append({'case': folder.name, 'slice': path.name,
                                 'source': str(path), 'output': str(saved_path),
                                 'error': str(error)})
        record = {
            'case': folder.name, 'z0': z0, 'z1_exclusive': z1,
            'original_width': width, 'original_height': height,
            'cropped_width': width, 'cropped_height': z1-z0,
            'expected_slices': len(files), 'saved_verified_slices': case_saved,
            'complete': case_saved == len(files), 'layer_file': str(label_path),
        }
        metadata.append(record)
        save_json(destination / 'crop_metadata.json', record)
        if index == 1 or index % 5 == 0 or index == len(plans):
            print(f'[{index}/{len(plans)}] {folder.name}: {width}x{height} -> '
                  f'{width}x{z1-z0}; 已保存并验证 {saved_count} 张; '
                  f'失败 {len(failures)} 张', flush=True)

    with (output / 'crop_coordinates.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    summary = {
        'case_count': len(plans), 'complete_cases': sum(r['complete'] for r in metadata),
        'expected_images': sum(len(p[1]) for p in plans),
        'saved_verified_images': saved_count, 'failed_images': len(failures),
        'cropped_height_min': min(r['cropped_height'] for r in metadata),
        'cropped_height_median': float(np.median([r['cropped_height'] for r in metadata])),
        'cropped_height_max': max(r['cropped_height'] for r in metadata),
        'elapsed_seconds': round(time.monotonic()-started, 2),
        'failures': failures,
    }
    save_json(output / 'crop_summary.json', summary)
    (output / 'README_裁剪说明.txt').write_text(
        f'OCTA-3mm {args.modality} 裁剪结果\n'
        f'原始 {args.modality}：{source}\n' +
        (f'直接复用参考 OCT 坐标：{coordinate_path}\n' if coordinate_path is not None else '') +
        f'按整卷 GT_Layers 的 ILM/BM 边界，上保留 {args.margin_top}、下保留 {args.margin_bottom} 像素。\n'
        '每例所有切片使用同一裁剪范围，保留 BMP 格式和原文件名。\n'
        '没有缩放、归一化或补边；输出高度随病例变化。\n'
        '坐标见 crop_coordinates.csv；区间为 Python 0 基 [z0,z1_exclusive)。\n'
        'OCT/OCTA 配对数据应共用同一病例坐标。原始数据没有修改。\n'
        '定位来自已有层标注，不是仅输入 OCT 的自动定位模型。\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
