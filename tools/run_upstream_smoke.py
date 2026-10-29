"""Run unmodified author's train.py and test.py on synthetic data, then record evidence."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from make_demo_data import make_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='smoke_synthetic')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / 'artifacts' / args.name
    out.mkdir(parents=True, exist_ok=False)
    data = out / 'data'
    make_data(data)
    common = ['--dataroot', str(data), '--name', args.name, '--gpu_ids', '0',
              '--load_size', '32', '--crop_size', '32', '--checkpoints_dir', str(out / 'checkpoint')]
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    start = time.perf_counter()
    def run(filename, arguments, log):
        command = [sys.executable, filename, *arguments]
        print('Running:', filename, flush=True)
        with (out / log).open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=root / 'upstream', stdout=stream, stderr=subprocess.STDOUT, env=env)
        if result.returncode:
            raise RuntimeError(f'{filename} failed; see {out / log}')
    run('train.py', common + ['--n_epochs', '1', '--num_threads', '0', '--print_freq', '1'], 'train.log')
    weights = sorted((out / 'checkpoint' / args.name).glob('best*_net_G.pth'))
    if not weights:
        raise RuntimeError('No best checkpoint saved')
    label = weights[-1].name.removesuffix('_net_G.pth')
    run('test.py', common + ['--epoch', label, '--results_dir', str(out / 'results')], 'test.log')
    predictions = list((out / 'results').rglob('*.npz'))
    assert len(predictions) == 1
    with np.load(predictions[0]) as prediction:
        assert set(prediction.files) == {'oct', 'octa', 'octa_gt'}
        assert prediction['octa'].shape == (32, 32, 32)
        assert np.isfinite(prediction['octa']).all()
        assert 0 <= prediction['octa'].min() <= prediction['octa'].max() <= 1
    import torch
    packages = ['torch', 'torchvision', 'monai', 'torchio', 'numpy', 'scikit-image', 'PyWavelets', 'matplotlib', 'opencv-python', 'lpips', 'clean-fid']
    report = {'status': 'passed', 'scope': 'Synthetic data only: 1 epoch / 2 optimizer updates / validation / checkpoint reload / 3-step test inference',
              'not_verified': ['OCTA-500 numerical reproduction', 'patient-specific vascular fidelity', 'full-size training memory'],
              'python': sys.version, 'gpu': torch.cuda.get_device_name(0), 'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
              'packages': {p: importlib.metadata.version(p) for p in packages},
              'commit': subprocess.check_output(['git', '-C', str(root / 'upstream'), 'rev-parse', 'HEAD'], text=True).strip(),
              'checkpoint_label': label, 'elapsed_seconds': time.perf_counter() - start,
              'prediction': str(predictions[0])}
    (out / 'verification.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
