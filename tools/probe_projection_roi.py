"""Demonstrate the projection indexing bug and verify the minimal fix with a full update."""
import json
from pathlib import Path
import subprocess
import sys


def worker(repository):
    import torch
    from types import SimpleNamespace
    sys.path.insert(0, str(repository))
    from models.FlowOCT_model import FlowOCT
    torch.manual_seed(2026)
    opt = SimpleNamespace(gpu_ids=[0], isTrain=True, checkpoints_dir='unused_probe', name='probe',
                          lr=2e-4, beta=0.5, continue_train=False)
    model = FlowOCT(opt)
    # Both modalities have a band away from row zero; vertical padding still yields y_min > 0.
    a = torch.zeros(1, 1, 32, 32, 32)
    b = torch.zeros_like(a)
    a[:, :, :, 14:18, 10:22] = 0.3
    b[:, :, :, 14:18, 10:22] = 0.8
    model.set_input({'A': a, 'B': b})
    model.forward()
    losses = model.get_current_loss()
    import math
    losses = {k: v if math.isfinite(v) else 'NaN' for k, v in losses.items()}
    return {'repository': str(repository), 'losses': losses,
            'total_finite': bool(torch.isfinite(model.loss)),
            'parameters_finite_after_step': all(bool(torch.isfinite(p).all()) for p in model.netG.parameters())}


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        print(json.dumps(worker(Path(sys.argv[2]))))
    else:
        root = Path(__file__).resolve().parents[1]
        records = []
        for name in ('upstream', 'working'):
            output = subprocess.check_output([sys.executable, __file__, '--worker', str(root / name)], text=True)
            records.append(json.loads(output.strip().splitlines()[-1]))
        assert not records[0]['total_finite']
        assert records[1]['total_finite'] and records[1]['parameters_finite_after_step']
        (root / 'artifacts' / 'projection_roi_probe.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
        print(json.dumps(records, indent=2))
