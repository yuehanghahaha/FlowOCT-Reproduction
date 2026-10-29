"""Check workflow ordering and preservation with a fake training CLI, without a GPU."""
import csv
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT = ROOT / "FlowOCT_test200_then_train400.sh"
BASH = Path(r"D:\software\Git\bin\bash.exe")
CORE = [ROOT / "flowoct_reproduction" / p for p in
        ("flowoct/model.py", "flowoct/runner.py", "flowoct/data.py",
         "flowoct/metrics.py", "configs/3m.json")]

FAKE_RUN = r"""
import argparse, csv, hashlib, json, os
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("command")
for name in ("config", "device", "checkpoint", "split", "output", "predictions", "resume", "epochs"):
    p.add_argument("--" + name)
a = p.parse_args()
with Path("calls.jsonl").open("a", encoding="utf-8") as f:
    f.write(json.dumps(vars(a)) + "\n")
if a.command == "predict":
    if os.environ.get("FAIL_PREDICT") == "1":
        raise SystemExit(7)
    out = Path(a.output)
    out.mkdir(parents=True)
    checkpoint = json.loads(Path(a.checkpoint).read_text())
    digest = hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    records = []
    for i in range(50):
        case = str(10301 + i)
        (out / (case + ".npz")).write_bytes(b"workflow fixture, not a medical image")
        records.append({"case_id": case})
    info = {"split": "test", "limited_diagnostic": False, "records": records,
            "checkpoint_sha256": digest, "checkpoint_epoch": checkpoint["epoch"]}
    (out / "prediction.json").write_text(json.dumps(info), encoding="utf-8")
elif a.command == "evaluate":
    info = json.loads((Path(a.predictions) / "prediction.json").read_text())
    out = Path(a.output)
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps({
        "cases": len(info["records"]), "protocol": {"checkpoint_sha256": info["checkpoint_sha256"]}}))
    with (out / "per_case.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id"])
        w.writeheader()
        w.writerows(info["records"])
elif a.command == "train":
    assert a.epochs == "400"
    output = Path(a.output).resolve()
    assert Path(a.resume).resolve() == output / "latest.pt"
    latest = json.loads(Path(a.resume).read_text())
    assert latest["epoch"] == 200
    cfg = json.loads(Path(a.config).read_text())
    cfg["epochs"] = 400
    (output / "config.json").write_text(json.dumps(cfg))
    latest["epoch"] = 201
    latest["global_step"] = 28140
    (output / "latest.pt").write_text(json.dumps(latest))
"""

def hashes(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*") if p.is_file()}

def exercise(kind):
    with tempfile.TemporaryDirectory(prefix="flowoct_workflow_", dir=HERE) as folder:
        work = Path(folder).resolve()
        assert work.is_relative_to(HERE.resolve())
        project = work / "flowoct_reproduction_a100"
        base = project / "runs/flowoct_3m_200e"
        base.mkdir(parents=True)
        (work / "logs").mkdir()
        (work / "logs/flowoct_3m_train.log").write_text("original 200 epoch fixture\n")
        cfg = {"epochs": 200, "fov": "3M", "network_chunk": 0, "diagnostic_limits": None}
        (base / "config.json").write_text(json.dumps(cfg))
        best = {"format": "flowoct-reproduction-v1", "epoch": 193,
                "global_step": 27020, "best_psnr": 29.741366907744993}
        latest = dict(best, epoch=200, global_step=28000)
        if kind == "incomplete_200":
            latest["epoch"] = 199
        (base / "best.pt").write_text(json.dumps(best))
        (base / "latest.pt").write_text(json.dumps(latest))
        for name in ("epochs.jsonl", "manifest.csv", "environment.json", "training_scope.json"):
            (base / name).write_text("fixture for " + name)
        history = [{"epoch": e, "validation": {"metrics": {"volume_psnr": {
            "mean": best["best_psnr"] if e == 193 else 29.0}}}} for e in range(1, 201)]
        (base / "epochs.jsonl").write_text("\n".join(json.dumps(row) for row in history) + "\n")
        (project / "torch.py").write_text(
            "import json\n__version__ = '2.5.1'\ndef load(path, **kwargs):\n    return json.loads(path.read_text())\n")
        for module, version in (("torchio", "0.20.4"), ("monai", "1.4.0"),
                                ("numpy", "1.26.4"), ("skimage", "0.25.0")):
            (project / (module + ".py")).write_text("__version__ = " + repr(version) + "\n")
        if kind == "missing_dependency":
            (project / "torchio.py").write_text("raise ModuleNotFoundError(\"No module named 'torchio'\")\n")
        (project / "run.py").write_text(FAKE_RUN)
        if kind == "existing_output":
            (project / "results/flowoct_3m_best_after200").mkdir(parents=True)
        bindir = work / "bin"
        bindir.mkdir()
        python = Path(sys.executable).as_posix()
        (bindir / "python").write_text('#!/usr/bin/env bash\nprintf "Wrong ambient Python was used" >&2\nexit 93\n')
        selected_python = bindir / "flowoct-python"
        selected_python.write_text('#!/usr/bin/env bash\nexec "' + python + '" -X utf8 "$@"\n')
        env = dict(os.environ)
        env["PATH"] = str(bindir) + os.pathsep + env["PATH"]
        env["FLOWOCT_PYTHON"] = selected_python.as_posix()
        if kind == "missing_interpreter":
            env["FLOWOCT_PYTHON"] = (bindir / "missing-python").as_posix()
        env["PYTHONUTF8"] = "1"
        if kind == "predict_failure":
            env["FAIL_PREDICT"] = "1"
        previous_log = work / "logs/flowoct_3m_best_after200_test.log"
        previous_content = "Previous failure: No module named 'torchio'\n"
        if kind in ("retry_failed_log", "missing_dependency"):
            previous_log.write_text(previous_content)
        before = hashes(base)
        result = subprocess.run([str(BASH), SCRIPT.as_posix()], cwd=project, env=env,
                                capture_output=True, text=True, encoding="utf-8")
        assert hashes(base) == before, "Original baseline files were changed"
        callsfile = project / "calls.jsonl"
        calls = [json.loads(x) for x in callsfile.read_text().splitlines()] if callsfile.exists() else []
        archive = work / "uploads/flowoct_3m_best_after200_review.tar"
        extended = project / "runs/flowoct_3m_400e_from200"
        if kind in ("success", "retry_failed_log"):
            assert result.returncode == 0, result.stdout + result.stderr
            assert [r["command"] for r in calls] == ["predict", "evaluate", "train"]
            assert calls[0]["split"] == "test"
            assert calls[2]["epochs"] == "400"
            assert calls[2]["resume"] == "runs/flowoct_3m_400e_from200/latest.pt"
            assert json.loads((extended / "config.json").read_text())["epochs"] == 400
            assert json.loads((extended / "latest.pt").read_text())["epoch"] == 201
            assert archive.exists() and Path(str(archive) + ".sha256").exists()
            with tarfile.open(archive) as bundle:
                names = bundle.getnames()
                assert len([n for n in names if n.endswith(".npz")]) == 50
                for name in ("results/flowoct_3m_best_after200/prediction.json",
                             "evaluations/flowoct_3m_best_after200/summary.json",
                             "evaluations/flowoct_3m_best_after200/per_case.csv",
                             "runs/flowoct_3m_200e/best.pt",
                             "logs/flowoct_3m_train.log", "logs/flowoct_3m_best_after200_test.log"):
                    assert name in names, name
                archived_cfg = json.load(bundle.extractfile("runs/flowoct_3m_200e/config.json"))
                assert archived_cfg["epochs"] == 200
                archived_best = json.load(bundle.extractfile("runs/flowoct_3m_200e/best.pt"))
                assert archived_best["epoch"] == 193
            if kind == "retry_failed_log":
                backups = list((work / "logs").glob("flowoct_3m_best_after200_test.log.previous_*"))
                assert len(backups) == 1 and backups[0].read_text() == previous_content
                assert previous_log.read_text() != previous_content
        else:
            assert result.returncode != 0
            assert not archive.exists() and not extended.exists()
            expected = ["predict"] if kind == "predict_failure" else []
            assert [r["command"] for r in calls] == expected
            if kind == "missing_dependency":
                assert previous_log.read_text() == previous_content
                assert not list((work / "logs").glob("*.previous_*"))
        return {"case": kind, "passed": True, "returncode": result.returncode,
                "original_baseline_unchanged": True, "commands": [r["command"] for r in calls]}

def main():
    assert b"\r" not in SCRIPT.read_bytes(), "Server script must use LF line endings"
    assert SCRIPT.read_text(encoding="utf-8").isascii(), "Console messages should work in ASCII terminals"
    core_before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in CORE}
    cases = [exercise(k) for k in ("success", "predict_failure", "existing_output", "incomplete_200",
                                  "retry_failed_log", "missing_dependency", "missing_interpreter")]
    core_after = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in CORE}
    assert core_before == core_after
    result = {
        "scope": "Local Bash workflow checks using a fake CLI and fake checkpoints; no GPU inference or training.",
        "environment_regression": "Ambient python intentionally fails; every successful invocation must use the explicitly selected interpreter.",
        "cases": cases, "core_sha256": core_after,
        "script_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
    }
    (HERE / "transfer_workflow_verification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

