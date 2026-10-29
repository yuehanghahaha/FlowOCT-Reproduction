"""Prepare immutable paired OCTA-500 volumes and portable shared experiment manifests.

Run from the project root: python -m tools.prepare_common_octa500 --source ... --output ...
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import threading
import time
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from fm3d.data import COUNTS, validate_manifest


SCHEMA = 'octa500-common-v1'
FOVS = {'3M': ('OCTA_3mm', 10301, 10500, 304), '6M': ('OCTA_6mm', 10001, 10300, 400)}
QA_LOCK = threading.Lock()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha_array(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def read_metadata(path, expected):
    """Read the simple official XLSX sheet without evaluating formulas."""
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(path) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            root = ET.fromstring(archive.read('xl/sharedStrings.xml'))
            strings = [''.join(x.itertext()) for x in root.findall('m:si', ns)]
        root = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
        rows = {}
        for element in root.findall('m:sheetData/m:row', ns):
            cells = {}
            for cell in element.findall('m:c', ns):
                column = ''.join(c for c in cell.attrib['r'] if c.isalpha())
                value = cell.find('m:v', ns)
                value = value.text if value is not None else ''
                if cell.get('t') == 's':
                    value = strings[int(value)]
                elif cell.get('t') == 'inlineStr':
                    value = ''.join(cell.find('m:is', ns).itertext())
                cells[column] = value.strip()
            cid = cells.get('A', '')
            if not cid or cid == 'ID':
                continue
            if cid not in expected or cid in rows:
                raise ValueError(f'Unexpected/duplicate metadata ID {cid}: {path}')
            if not cells.get('E'):
                raise ValueError(f'Missing disease label: {cid}')
            rows[cid] = dict(case_id=cid, patient_id=cid, sex=cells.get('B', ''),
                             eye=cells.get('C', ''), age=cells.get('D', ''), disease=cells['E'])
    if set(rows) != set(expected):
        raise ValueError(f'Metadata cases differ from required IDs: {path}')
    return rows


def stratified_assign(rows, fov, seed):
    """Exact totals and proportional quotas, preserving disease coverage when feasible."""
    rng = random.Random(seed)
    groups = {}
    for row in sorted(rows, key=lambda r: r['case_id']):
        groups.setdefault(row['disease'], []).append(row)
    groups = dict(sorted(groups.items()))
    for group in groups.values():
        rng.shuffle(group)
    for phase, reserve in (('validation', 2), ('test', 1), ('train', 0)):
        total = sum(len(g) for g in groups.values())
        count = COUNTS[fov][phase]
        exact = {d: len(g) * count / total for d, g in groups.items()}
        lower = {d: int(len(g) >= reserve + 1) for d, g in groups.items()}
        upper = {d: max(0, len(g) - reserve) if len(g) >= reserve + 1 else len(g) for d, g in groups.items()}
        if sum(lower.values()) > count or sum(upper.values()) < count:
            raise ValueError('Cannot cover diseases with these split counts; review grouped allocation')
        quota = {d: min(upper[d], max(lower[d], math.floor(n))) for d, n in exact.items()}
        ties = {d: rng.random() for d in groups}
        while sum(quota.values()) < count:
            d = max((d for d in groups if quota[d] < upper[d]), key=lambda d: (exact[d] - quota[d], ties[d]))
            quota[d] += 1
        while sum(quota.values()) > count:
            d = max((d for d in groups if quota[d] > lower[d]), key=lambda d: (quota[d] - exact[d], ties[d]))
            quota[d] -= 1
        for d, group in groups.items():
            for row in group[:quota[d]]:
                row['split'] = phase
            groups[d] = group[quota[d]:]
    if any(groups.values()):
        raise AssertionError('Unassigned cases')


def read_case(source, row):
    subdir, _, _, width = FOVS[row['fov']]
    arrays, details = {}, {}
    expected = {f'{i}.bmp' for i in range(1, width + 1)}
    for modality, key in [('OCT', 'oct'), ('OCTA', 'octa')]:
        folder = source / subdir / modality / row['case_id']
        if {p.name for p in folder.glob('*.bmp')} != expected:
            raise ValueError(f'Missing or unexpected B-scan files: {folder}')
        array = np.empty((640, width, width), dtype=np.uint8)
        digest = hashlib.sha256()
        for i in range(1, width + 1):
            name = f'{i}.bmp'
            content = (folder / name).read_bytes()
            digest.update(name.encode('ascii') + b'\0' + content)
            with Image.open(io.BytesIO(content)) as image:
                if image.mode != 'L' or image.size != (width, 640):
                    raise ValueError(f'Unexpected image format: {folder / name}')
                array[:, i - 1, :] = np.asarray(image)
        if array.min() == array.max():
            raise ValueError(f'Constant volume: {folder}')
        arrays[key] = array
        details[key] = {'bmp_count': width, 'source_bmp_sha256': digest.hexdigest(),
                        'array_sha256': sha_array(array), 'min': int(array.min()), 'max': int(array.max())}
    return arrays, details


def z256(array):
    # Only Z changes: apply align_corners=False linear interpolation plane by plane.
    # This avoids allocating a complete float32 640-depth source volume.
    resized = np.empty((256, *array.shape[1:]), dtype=np.float32)
    for z in range(256):
        coordinate = (z + 0.5) * 640 / 256 - 0.5
        lo = math.floor(coordinate)
        fraction = coordinate - lo
        resized[z] = array[lo].astype(np.float32) * (1 - fraction) + array[lo + 1].astype(np.float32) * fraction
    # Independent PyTorch interpolation on deterministic columns, across ALL output depths.
    ys = np.linspace(0, array.shape[1] - 1, 17, dtype=int)
    xs = np.roll(np.linspace(0, array.shape[2] - 1, 17, dtype=int), 5)
    reference = torch.from_numpy(array[:, ys, xs].T.copy()).float()[None]
    expected = F.interpolate(reference, size=256, mode='linear', align_corners=False)[0].T.numpy()
    np.testing.assert_array_equal(resized[:, ys, xs], expected)
    return resized


def save_npz_verified(path, arrays, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.npz.tmp')
    with ZipFile(tmp, 'w', ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for key, array in {**arrays, **extra}.items():
            with archive.open(key + '.npy', 'w', force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
    with np.load(tmp, allow_pickle=False) as loaded:
        for key, array in arrays.items():
            actual = loaded[key]
            if actual.dtype != array.dtype or not np.array_equal(actual, array):
                raise AssertionError(f'NPZ roundtrip mismatch: {path}, {key}')
            del actual
    tmp.replace(path)
    return {'path': path.as_posix(), 'bytes': path.stat().st_size, 'sha256': sha_file(path)}


def qa_images(root, row, native, prepared):
    """Fixed display ranges; display-only PNGs, never training input."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cid = row['case_id']
    y = native['oct'].shape[1] // 2
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for index, key in enumerate(('oct', 'octa')):
        for col, (array, title) in enumerate((
            (native[key][:, y, :], f'{key.upper()} native B-scan'),
            (prepared[key][:, y, :], f'{key.upper()} Z256 B-scan'),
            (prepared[key].mean(axis=0), f'{key.upper()} Z-mean (fixed 0..80 display)'))):
            axes[index, col].imshow(array, cmap='gray', vmin=0, vmax=80 if col == 2 else 255, aspect='auto')
            axes[index, col].set_title(title)
            axes[index, col].set_xlabel('X')
            axes[index, col].set_ylabel('Y' if col == 2 else 'Z')
    fig.suptitle(f'{row["fov"]} / {cid} / {row["disease"]} / {row["split"]}; display only')
    fig.tight_layout()
    path = root / 'qa' / f'{row["fov"]}_{cid}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def process_case(source, output, row, qa=False):
    started = time.time()
    arrays, details = read_case(source, row)
    extra = {'axis_order': np.array('ZYX'), 'intensity_scale': np.array(255),
             'case_id': np.array(row['case_id']), 'patient_id': np.array(row['patient_id'])}
    native = output / 'native' / row['fov'] / f'{row["case_id"]}.npz'
    record = dict(row)
    record['source'] = details
    record['native'] = save_npz_verified(native, arrays, extra)
    record['native']['path'] = native.relative_to(output).as_posix()
    prepared = {k: z256(v) for k, v in arrays.items()}
    target = output / 'z256' / row['fov'] / f'{row["case_id"]}.npz'
    record['z256'] = save_npz_verified(target, prepared, extra)
    record['z256']['path'] = target.relative_to(output).as_posix()
    record['z256']['array_sha256'] = {k: sha_array(v) for k, v in prepared.items()}
    record['z256']['shape'] = list(prepared['oct'].shape)
    record['native']['shape'] = list(arrays['oct'].shape)
    record['native']['dtype'] = 'uint8'
    record['z256']['dtype'] = 'float32'
    record['seconds'] = time.time() - started
    if qa:
        with QA_LOCK:
            qa_images(output, row, arrays, prepared)
    write_json(output / 'records' / f'{row["case_id"]}.json', record)
    return record


def protect_duplicate_groups(rows, records):
    """Keep exact content duplicates in train, exchanging same-disease singletons.

    Identity remains the public ID: content equality does not prove patient identity.
    Fail closed if the simple exchange cannot preserve all disease/split counts.
    """
    parent = {r['case_id']: r['case_id'] for r in rows}
    def root(cid):
        while parent[cid] != cid:
            cid = parent[cid]
        return cid
    for modality in ('oct', 'octa'):
        seen = {}
        for record in sorted(records, key=lambda r: r['case_id']):
            cid, digest = record['case_id'], record['source'][modality]['array_sha256']
            if digest in seen:
                parent[root(cid)] = root(seen[digest])
            seen[digest] = cid
    groups = {}
    for row in rows:
        row['leakage_group'] = root(row['case_id'])
        groups.setdefault(row['leakage_group'], []).append(row)
    duplicates, exchanges = [], []
    for gid, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        duplicates.append({'leakage_group': gid, 'case_ids': sorted(r['case_id'] for r in group),
                           'metadata': [{k: r[k] for k in ('case_id', 'disease', 'sex', 'eye', 'age')} for r in group]})
        for row in sorted(group, key=lambda r: r['case_id']):
            if row['split'] == 'train':
                continue
            candidates = [r for r in rows if r['split'] == 'train' and r['fov'] == row['fov']
                          and r['disease'] == row['disease'] and len(groups[r['leakage_group']]) == 1]
            if not candidates:
                raise ValueError('Cannot preserve quotas while grouping duplicate images; review split')
            replacement = max(candidates, key=lambda r: r['case_id'])
            old_split = row['split']
            row['split'], replacement['split'] = 'train', old_split
            exchanges.append({'to_train': row['case_id'], 'from_train': replacement['case_id'], 'other_split': old_split})
    for group in groups.values():
        if len({r['split'] for r in group}) != 1:
            raise AssertionError('Duplicate content crosses splits')
    return {'policy': 'Exact OCT or OCTA pixel identity links cases; duplicate groups kept in train; deterministic largest-ID same-disease singleton exchanges preserve quotas',
            'groups': duplicates, 'exchanges': exchanges, 'unique_content_groups': len(groups),
            'identity_limit': 'Image equality does not establish patient identity; metadata conflicts retained, not corrected',
            'weighting': 'All public IDs retained; repeated training images are sampled once per ID and therefore receive extra weight'}


def write_manifests(output, rows):
    for fov in FOVS:
        for representation in ('native', 'z256'):
            path = output / f'split_{fov}_{representation}.csv'
            selected = [r for r in rows if r['fov'] == fov]
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['case_id', 'patient_id', 'path', 'split', 'fov', 'disease', 'sex', 'eye', 'age', 'leakage_group'])
                writer.writeheader()
                for row in selected:
                    writer.writerow({**row, 'path': f'{representation}/{fov}/{row["case_id"]}.npz'})
            validate_manifest([{**row, 'path': str(output / representation / fov / f'{row["case_id"]}.npz')} for row in selected], fov)


def copy_support(source, output):
    records = []
    sources = [(source / 'Label', output / 'labels_native')]
    for fov, (subdir, _, _, _) in FOVS.items():
        sources.append((source / subdir / 'Projection Maps', output / 'official_projections' / fov))
        src = source / subdir / 'Text labels.xlsx'
        dst = output / 'metadata' / f'{fov}_text_labels.xlsx'
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if sha_file(src) != sha_file(dst):
            raise AssertionError('Metadata copy mismatch')
        records.append({'path': dst.relative_to(output).as_posix(), 'sha256': sha_file(dst), 'bytes': dst.stat().st_size})
    for srcroot, dstroot in sources:
        for src in sorted(srcroot.rglob('*')):
            if not src.is_file():
                continue
            dst = dstroot / src.relative_to(srcroot)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            digest = sha_file(src)
            if sha_file(dst) != digest:
                raise AssertionError(f'Support file mismatch: {src}')
            records.append({'path': dst.relative_to(output).as_posix(), 'sha256': digest, 'bytes': dst.stat().st_size})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--limit', type=int, help='Pilot only; do not publish final manifests')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError('Output must be outside the original dataset')
    config = {'schema': SCHEMA, 'source': str(source), 'seed': args.seed, 'counts': COUNTS,
              'axis_order': 'ZYX', 'intensity_scale': 255, 'native_dtype': 'uint8', 'z256_dtype': 'float32',
              'resize': 'NumPy memory-bounded Z-only linear 640->256, equivalent to trilinear with unchanged Y/X; align_corners=False; checked against PyTorch; no antialias filter',
              'intensity': 'No normalization, no clipping, no per-volume minmax; values remain 0..255',
              'roi': 'Full FOV, no annotation-guided crop, denoise, flattening or new registration',
              'split': 'Disease-proportional bounded quotas, validation then test then remaining train; minimum one per disease/split when feasible; seeded shuffle; own protocol, not paper ID lists',
              'patient_id_basis': 'Official dataset paper section 3.1: 500 subjects, one eye each; public ID used as subject ID, no private identity linkage available',
              'torch': torch.__version__, 'numpy': np.__version__}
    if output.exists() and any(output.iterdir()):
        if not args.resume or json.loads((output / 'protocol.json').read_text(encoding='utf-8')) != config:
            raise ValueError('Nonempty output requires --resume and identical protocol')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'protocol.json', config)
    torch.set_num_threads(2)
    rows = []
    for fov, (subdir, first, last, width) in FOVS.items():
        expected = {str(i) for i in range(first, last + 1)}
        for modality in ('OCT', 'OCTA'):
            found = {p.name for p in (source / subdir / modality).iterdir() if p.is_dir()}
            if found != expected:
                raise ValueError(f'Unexpected cases: {fov}/{modality}')
        metadata = read_metadata(source / subdir / 'Text labels.xlsx', expected)
        current = [{**metadata[cid], 'fov': fov} for cid in sorted(expected)]
        stratified_assign(current, fov, args.seed)
        rows.extend(current)
    write_json(output / 'assignment_provisional.json', rows)
    # QA examples chosen from metadata only: first ID per disease per FOV.
    qa_ids, seen = set(), set()
    for row in rows:
        key = (row['fov'], row['disease'])
        if key not in seen:
            qa_ids.add(row['case_id'])
            seen.add(key)
    records, began = [], time.time()
    if args.workers < 1 or args.workers > 2:
        raise ValueError('Use 1 or 2 workers to bound memory')
    def prepare(row):
        record_path = output / 'records' / f'{row["case_id"]}.json'
        if record_path.exists() and args.resume:
            record = json.loads(record_path.read_text(encoding='utf-8'))
            if any(record[k] != row[k] for k in row if k != 'split'):
                raise ValueError('Resume case metadata changed')
            for kind in ('native', 'z256'):
                if sha_file(output / record[kind]['path']) != record[kind]['sha256']:
                    raise ValueError(f'Resume output checksum mismatch: {row["case_id"]}')
        else:
            if shutil.disk_usage(output).free < 5 * 1024 ** 3:
                raise OSError('Less than 5 GiB free; stopping without changing source data')
            record = process_case(source, output, row, qa=row['case_id'] in qa_ids)
        return record
    selected = rows[:args.limit] if args.limit else rows
    # Submit at most workers cases at once; matplotlib rendering uses a lock.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(selected), args.workers):
            batch = selected[start:start + args.workers]
            futures = [pool.submit(prepare, row) for row in batch]
            for future in futures:
                record = future.result()
                records.append(record)
                progress = {'completed': len(records), 'total': len(rows), 'last_case': record['case_id'], 'fov': record['fov'],
                            'elapsed_seconds': time.time() - began,
                            'volume_bytes': sum(r[k]['bytes'] for r in records for k in ('native', 'z256'))}
                write_json(output / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    if len(records) != len(rows):
        print('PILOT ONLY: no final training manifests published', flush=True)
        return
    duplicate_report = protect_duplicate_groups(rows, records)
    write_json(output / 'duplicate_audit.json', duplicate_report)
    write_json(output / 'assignment.json', rows)
    by_id = {r['case_id']: r for r in rows}
    for record in records:
        record.update(by_id[record['case_id']])
        write_json(output / 'records' / f'{record["case_id"]}.json', record)
    write_manifests(output, rows)
    support = copy_support(source, output)
    distribution = {}
    for fov in FOVS:
        distribution[fov] = {}
        for phase in COUNTS[fov]:
            diseases = {}
            for row in rows:
                if row['fov'] == fov and row['split'] == phase:
                    diseases[row['disease']] = diseases.get(row['disease'], 0) + 1
            distribution[fov][phase] = diseases
    audit = {'status': 'complete', 'cases': len(rows), 'distribution': distribution,
                                      'duplicate_audit': duplicate_report,
                                      'volumes': records, 'support_files': support,
                                      'verification': ['all source BMP names/modes/shapes checked', 'all NPZ arrays read back and compared exactly',
                                                       '17 spatial columns across all 256 output depths independently checked against PyTorch per modality/case',
                                                       'exact duplicate source pixels grouped within train, none across splits', 'support files copied and SHA256 matched'],
                                      'limits': ['registration inherited from dataset, not independently re-estimated',
                                                 'labels/official projections remain in their original grid and display conventions',
                                                 'duplicate public IDs retained in train: case counts are not counts of unique images',
                                                 'rare diseases may not occur in every split; no model-performance-based selection']}
    with (output / 'SHA256SUMS.txt').open('w', encoding='utf-8', newline='\n') as stream:
        for record in records:
            for kind in ('native', 'z256'):
                stream.write(f'{record[kind]["sha256"]}  {record[kind]["path"]}\n')
        for record in support:
            stream.write(f'{record["sha256"]}  {record["path"]}\n')
        for file in sorted(output.glob('split_*.csv')):
            stream.write(f'{sha_file(file)}  {file.name}\n')
        for name in ('protocol.json', 'assignment.json', 'duplicate_audit.json'):
            stream.write(f'{sha_file(output / name)}  {name}\n')
    write_json(output / 'audit.json', audit)
    print('COMPLETE: 500 cases, native + Z256, manifests and original support data verified', flush=True)


if __name__ == '__main__':
    main()
