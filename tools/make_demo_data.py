"""Create synthetic paired volumes. These are NOT OCTA-500 or medical data."""
import argparse
import json
from pathlib import Path
import numpy as np


def make_data(root, size=32, seed=2026):
    root = Path(root)
    rng = np.random.default_rng(seed)
    d, h, w = np.meshgrid(*[np.linspace(-1, 1, size)] * 3, indexing='ij')
    for phase, count in [('train', 2), ('validation', 1), ('test', 1)]:
        folder = root / phase
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            path = folder / f'synthetic_{phase}_{i:03d}.npz'
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite: {path}')
            shift = rng.uniform(-0.2, 0.2)
            vessel = np.exp(-((h - 0.25 * np.sin(4 * d) - shift) ** 2 + w ** 2) / 0.035)
            layers = 0.35 + 0.2 * np.cos(8 * w) + 0.15 * np.exp(-h ** 2 / 0.4)
            oct_volume = np.clip(layers + 0.12 * vessel + rng.normal(0, 0.025, d.shape), 0, 1)
            octa_volume = np.clip(vessel + rng.uniform(0, 0.025, d.shape), 0, 1)
            np.savez_compressed(path, oct=oct_volume.astype('float32'), octa=octa_volume.astype('float32'))
    (root / 'SYNTHETIC_ONLY.json').write_text(json.dumps({
        'purpose': 'Pipeline smoke test only; no scientific or medical conclusions',
        'seed': seed, 'shape': [size] * 3, 'axis_semantics': 'synthetic index axes only',
    }, indent=2), encoding='utf-8')
    print(f'Created synthetic-only dataset: {root.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    make_data(args.output, args.size, args.seed)
