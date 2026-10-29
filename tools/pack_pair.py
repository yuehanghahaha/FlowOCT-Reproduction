"""Pack one OCT/OCTA pair; does not infer anatomical axes or perform registration."""
import argparse
import json
from pathlib import Path
import re
import numpy as np


def natural_key(path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', path.name)]


def read_volume(path):
    if path.is_dir():
        from PIL import Image
        files = sorted([p for p in path.iterdir() if p.suffix.lower() in ('.bmp', '.png', '.tif', '.tiff')], key=natural_key)
        if not files:
            raise ValueError(f'No supported slice images in {path}')
        arrays = []
        for p in files:
            with Image.open(p) as image:
                array = np.asarray(image)
            if array.ndim != 2:
                raise ValueError(f'Expected native single-channel image (no silent RGB conversion): {p}, {array.shape}')
            arrays.append(array)
        return np.stack(arrays), [p.name for p in files]
    if path.suffix.lower() == '.npy':
        return np.load(path, allow_pickle=False), []
    raise ValueError('Supported input: .npy 3D volume or a folder of native grayscale BMP/PNG/TIF slices. For MAT/NIfTI, inspect keys/affine first.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oct', type=Path, required=True)
    parser.add_argument('--octa', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--patient-id', required=True)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], required=True)
    parser.add_argument('--transpose', type=int, nargs=3, required=True, help='verified source-to-repository index permutation, e.g. 0 1 2')
    parser.add_argument('--axis-description', required=True, help='actual physical meaning of output axes, established from dataset metadata')
    args = parser.parse_args()
    if sorted(args.transpose) != [0, 1, 2]:
        raise ValueError('transpose must be a permutation of 0 1 2')
    if args.output.suffix != '.npz' or args.output.parent.name != args.split:
        raise ValueError('output must be an .npz file inside its declared split directory')
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('Refusing to overwrite output or metadata')
    a, names_a = read_volume(args.oct)
    b, names_b = read_volume(args.octa)
    if a.ndim != 3 or a.shape != b.shape:
        raise ValueError(f'Expected equal 3D volumes: {a.shape} / {b.shape}')
    if names_a != names_b:
        raise ValueError('Slice filenames differ; explicitly audit slice correspondence before packing')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or a.min() == a.max() or b.min() == b.max():
        raise ValueError('Nonfinite or constant volume')
    metadata = {'patient_id': args.patient_id, 'split': args.split, 'source_oct': str(args.oct.resolve()),
                'source_octa': str(args.octa.resolve()), 'source_shape': list(a.shape),
                'transpose': args.transpose, 'output_axis_description': args.axis_description,
                'slice_names': names_a, 'normalization': 'none; upstream performs crop min-max',
                'registration': 'not performed; must be verified by user'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, oct=np.transpose(a, args.transpose).astype('float32'),
                        octa=np.transpose(b, args.transpose).astype('float32'))
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
