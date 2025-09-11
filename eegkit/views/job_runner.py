# eegkit/service/job_runner.py
from dataclasses import fields
from pathlib import Path
import json
import subprocess
import shutil
import datetime
import uuid
import re


class JobRunner:
    def __init__(self, controller, jobs_root: Path, runner_module: str = "eegkit.views.job_runner_script"):
        self.controller = controller
        self.jobs_root = Path(jobs_root)
        self.jobs_root.mkdir(exist_ok=True, parents=True)
        self.runner_module = runner_module

    def schedule(self, group: str, key: str, dto, params_dto):
        job_dir = self._create_job_dir(group, key, getattr(dto, "task", "task"))
        spec_path = job_dir / "spec.json"
        self._write_spec(spec_path, group, key, dto, params_dto, job_dir)
        runner_path = self._write_runner(job_dir, spec_path)
        self._launch(job_dir, runner_path)
        return job_dir

    def _create_job_dir(self, group: str, key: str, task_name: str) -> Path:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dto_part = re.sub(r"[^A-Za-z0-9]+", "_", f"{group}_{key}_{task_name}")[:40]
        job_id = f"job_{ts}_{dto_part}_{uuid.uuid4().hex[:6]}"
        job_dir = self.jobs_root / job_id
        (job_dir / "figures").mkdir(parents=True, exist_ok=True)
        return job_dir

    def _write_spec(self, spec_path: Path, group: str, key: str, dto, params_dto, job_dir: Path):
        schema_cls_name = type(dto).__name__
        params_cls_name = type(params_dto).__name__ if params_dto else None
        data_dir = str(getattr(self.controller.subject_model, "_data_dir", "."))

        schema_kwargs = {f.name: getattr(dto, f.name) for f in fields(dto)}
        params_kwargs = (
            {f.name: getattr(params_dto, f.name) for f in fields(params_dto)}
            if params_dto
            else None
        )

        spec = {
            "group": group,
            "key": key,
            "schema_class": schema_cls_name,
            "schema_kwargs": schema_kwargs,
            "params_class": params_cls_name,
            "params_kwargs": params_kwargs,
            "data_dir": data_dir,
            "job_id": job_dir.name,
            "job_dir": str(job_dir),
        }
        spec_path.write_text(json.dumps(spec, indent=2))

    def _write_runner(self, job_dir: Path, spec_path: Path) -> Path:
        runner_path = job_dir / "run.py"
        code = f"""#!/usr/bin/env python3
import sys
from pathlib import Path
from importlib import import_module

SPEC_PATH = r"{spec_path.as_posix()}"
MODULE_NAME = "{self.runner_module}"

_candidates = [
    Path.cwd(),
    Path(__file__).resolve().parent,                  # job_dir
    Path(__file__).resolve().parent.parent,           # jobs_root
    Path(__file__).resolve().parent.parent.parent,    # repo root (likely)
]
for _p in _candidates:
    if (_p / "eegkit").exists():
        sys.path.insert(0, str(_p))
        break

try:
    mod = import_module(MODULE_NAME)
except Exception as e:
    print(f"[ERROR] Could not import '{{MODULE_NAME}}': {{e}}")
    sys.exit(1)

if not hasattr(mod, "main"):
    print(f"[ERROR] '{{MODULE_NAME}}' has no function 'main(spec_path)'.")
    sys.exit(1)

ret = mod.main(SPEC_PATH)
sys.exit(int(ret) if isinstance(ret, int) else 0)
"""
        runner_path.write_text(code)
        try:
            runner_path.chmod(0o755)
        except Exception:
            pass
        return runner_path


    def _launch(self, job_dir: Path, runner_path: Path):
        """
        Launch the job in a detached tmux session when available;
        otherwise fall back to a background subprocess.
        """
        tmux_path = shutil.which("tmux")
        session_name = job_dir.name

        if tmux_path:
            cmd = [
                tmux_path,
                "new-session", "-d", "-s", session_name,
                f"python {runner_path.as_posix()} 2>&1 | tee -a {(job_dir / 'job.log').as_posix()}"
            ]
            print("[DEBUG] tmux launch command:", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
                print(f"[INFO] Launched tmux session: {session_name}")
                print(f"[INFO] Job directory: {job_dir}")
                print("\n--- To inspect logs ---")
                print(f"tmux attach -t {session_name}")
                print("------------------------------------------------\n")
                return
            except Exception as e:
                print(f"[WARN] Failed to start tmux session ({e}). Running inline...")

        # fallback: non-blocking background process
        cmd = ["python", runner_path.as_posix()]
        print("[DEBUG] subprocess launch command:", " ".join(cmd))
        subprocess.Popen(cmd)
        print("[INFO] tmux not found. Running job inline (non-blocking).")
        print(f"[INFO] Job directory: {job_dir}")


