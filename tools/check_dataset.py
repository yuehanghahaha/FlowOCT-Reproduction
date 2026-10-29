"""Validate the npz contract and split independence before training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def check(root, manifest=None):
    root = Path(root)
    errors, records, content_seen = [], [], {}
    for phase in ('train', 'validation', 'test'):
        files = sorted((root / phase).rglob('*.npz'))
        if not files:
            errors.append(f'{phase}: no npz files')
        for path in files:
            try:
                with np.load(path, allow_pickle=False) as data:
                    if not {'oct', 'octa'} <= set(data.files):
                        raise ValueError('required keys: oct, octa')
                    a, b = data['oct'], data['octa']
                    if a.ndim != 3 or a.shape != b.shape or min(a.shape) < 7:
                        raise ValueError(f'expected equal 3D arrays, all axes >=7: {a.shape}, {b.shape}')
                    if not np.isfinite(a).all() or not np.isfinite(b).all():
                        raise ValueError('NaN or Inf')
                    if a.max() == a.min() or b.max() == b.min():
                        raise ValueError('constant array will cause upstream min-max division by zero')
                    digest = hashlib.sha256(a.tobytes() + b.tobytes()).hexdigest()
                    if digest in content_seen and content_seen[digest][0] != phase:
                        errors.append(f'cross-split identical content: {path} / {content_seen[digest][1]}')
                    content_seen[digest] = (phase, str(path))
                    records.append({'path': str(path.resolve()), 'split': phase, 'shape': list(a.shape),
                                    'oct_dtype': str(a.dtype), 'octa_dtype': str(b.dtype), 'sha256_arrays': digest})
            except Exception as exc:
                errors.append(f'{path}: {exc}')
    if manifest:
        import csv
        seen = {}
        with open(manifest, encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                if not row.get('patient_id') or row.get('split') not in ('train', 'validation', 'test'):
                    errors.append('manifest requires patient_id and valid split columns')
                    continue
                patient, phase = row['patient_id'], row['split']
                if patient in seen and seen[patient] != phase:
                    errors.append(f'patient leakage: {patient}')
                seen[patient] = phase
    report = {'root': str(root.resolve()), 'files': records, 'errors': errors,
              'limitation': 'Without a complete patient manifest, patient independence is NOT verified. Axis orientation and registration require manual verification.'}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return bool(errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    raise SystemExit(check(args.dataroot, args.manifest))
