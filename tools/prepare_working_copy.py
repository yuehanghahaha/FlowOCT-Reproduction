"""Copy upstream and apply exactly one verified ROI indexing fix."""
from pathlib import Path
import json
import shutil
import subprocess

root = Path(__file__).resolve().parents[1]
source, target = root / 'upstream', root / 'working'
if target.exists():
    raise FileExistsError(f'Refusing to overwrite {target}')
shutil.copytree(source, target, ignore=shutil.ignore_patterns('.git', '__pycache__'))
path = target / 'models' / 'FlowOCT_model.py'
text = path.read_text(encoding='utf-8')
old = 'fake_proj[ym:yM+1, xm:xM+1],\n            real_proj[ym:yM+1, xm:xM+1],'
new = 'fake_proj[:, ym:yM+1, xm:xM+1],\n            real_proj[:, ym:yM+1, xm:xM+1],'
if text.count(old) != 1:
    raise RuntimeError('Expected projection indexing source not found exactly once')
path.write_text(text.replace(old, new), encoding='utf-8')
(target / 'LOCAL_FIX.json').write_text(json.dumps({
    'upstream_commit': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
    'modification': 'Projection ROI indexing preserves channel axis; two expression changes only',
    'verification': '../artifacts/projection_roi_probe.json',
    'limitations': 'All other upstream behavior remains; not a reconstruction of paper-only components',
}, indent=2), encoding='utf-8')
print(target)
