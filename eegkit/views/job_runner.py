"""Background job runner utilities for launching EEG tasks out-of-band."""
import datetime
import json
import os
import re
import shutil
import subprocess
import uuid
import sys
import shlex
from dataclasses import fields
from pathlib import Path


class JobRunner:
    """Create job specs and launch them via tmux or subprocess."""

    def __init__(
        self,
        controller,
        jobs_root: Path,
        runner_module: str = "eegkit.views.job_runner_script",
        queue_enabled: bool | None = None,
    ):
        """Initialize the job runner with a controller and jobs root directory."""
        self.controller = controller
        self.jobs_root = Path(jobs_root)
        self.jobs_root.mkdir(exist_ok=True, parents=True)
        self.runner_module = runner_module
        self._runs_log_path = self.jobs_root / "runs.log"
        # Use the same Python executable as the current process to avoid env issues in tmux
        self.python_exec = sys.executable or "python"
        env_default = os.getenv("EEG_JOB_QUEUE", "1").strip().lower() not in {"0", "false", "no"}
        self.queue_enabled = env_default if queue_enabled is None else bool(queue_enabled)
        self.queue_dir = self.jobs_root / ".queue"
        self.queue_dir.mkdir(exist_ok=True, parents=True)

    def schedule(self, group: str, key: str, dto, params_dto):
        """Create a job directory, persist its spec, and launch it."""
        job_dir = self._create_job_dir(group, key, getattr(dto, "task", "task"))
        spec_path = job_dir / "spec.json"
        self._write_spec(spec_path, group, key, dto, params_dto, job_dir)
        runner_path = self._write_runner(job_dir, spec_path)
        self._log_run(job_dir, group, key, dto, params_dto)
        if self.queue_enabled:
            self._enqueue(job_dir, runner_path)
            self._start_queue_worker()
        else:
            self._launch(job_dir, runner_path)

    def _safe_name(self, value: str, default: str = "na") -> str:
        """Return a filesystem-safe token from an arbitrary string."""
        if not value:
            return default
        value = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
        return value or default

    def _create_job_dir(self, group: str, key: str, task_name: str) -> Path:
        """Create and return a unique job directory path for this run."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_task = self._safe_name(task_name, "task")
        safe_key = self._safe_name(key, "plot")
        base_dir = self.jobs_root / safe_task / safe_key
        base_dir.mkdir(parents=True, exist_ok=True)
        job_id = f"{ts}_{uuid.uuid4().hex[:6]}"
        job_dir = base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def _write_spec(self, spec_path: Path, group: str, key: str, dto, params_dto, job_dir: Path):
        """Serialize the execution spec to JSON on disk."""
        schema_cls_name = type(dto).__name__
        params_cls_name = type(params_dto).__name__ if params_dto else None
        data_dir = self.controller.subject_model.data_dir

        schema_kwargs = {f.name: getattr(dto, f.name) for f in fields(dto)}
        params_kwargs = (
            {f.name: getattr(params_dto, f.name) for f in fields(params_dto)}
            if params_dto
            else None
        )

        spec = {
            "spec_version": 1,
            "group": group,
            "key": key,
            "schema_class": schema_cls_name,
            "schema_kwargs": schema_kwargs,
            "params_class": params_cls_name,
            "params_kwargs": params_kwargs,
            "data_dir": data_dir,
            "job_id": job_dir.name,
            "job_dir": str(job_dir),
            "created_utc": datetime.datetime.now().isoformat() + "Z",
        }
        spec_path.write_text(json.dumps(spec, indent=2))

    def _write_runner(self, job_dir: Path, spec_path: Path) -> Path:
        """Write a small bootstrap script that imports and invokes the runner module."""
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
    Path(__file__).resolve().parent.parent,           # <plot_type>
    Path(__file__).resolve().parent.parent.parent,    # <task>
    Path(__file__).resolve().parent.parent.parent.parent,  # jobs root
]
for _p in _candidates:
    if (_p / "eegkit").exists():
        if str(_p) not in sys.path:
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

    def _log_run(self, job_dir: Path, group: str, key: str, dto, params_dto):
        """Append a JSON line describing the scheduled run to runs.log."""
        entry = {
            "timestamp": datetime.datetime.now().isoformat() + "Z",
            "job_id": job_dir.name,
            "job_dir": str(job_dir),
            "group": group,
            "key": key,
            "task": getattr(dto, 'task', None),
            "subject": getattr(dto, 'subject', None),
            "run": getattr(dto, 'run', None),
            "schema_class": type(dto).__name__,
            "params_class": type(params_dto).__name__ if params_dto else None,
        }
        line = json.dumps(entry, sort_keys=True)
        try:
            with self._runs_log_path.open('a') as f:
                f.write(line + '\n')
        except Exception:
            pass

    def _launch(self, job_dir: Path, runner_path: Path):
        """Launch via tmux when available, otherwise a background subprocess."""
        tmux_path = shutil.which("tmux")
        session_name = f"{job_dir.parent.name}_{job_dir.name}"[:48]
        log_path = job_dir / 'job.log'

        if tmux_path:
            shell_cmd = f"{shlex.quote(self.python_exec)} {shlex.quote(str(runner_path))} 2>&1 | tee -a {shlex.quote(str(log_path))}"
            cmd = [
                tmux_path,
                "new-session", "-d", "-s", session_name,
                shell_cmd,
            ]
            print("[DEBUG] tmux launch command:", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
                (job_dir / 'session.txt').write_text(session_name)
                self._append_session_log(session_name, job_dir)
                print(f"[INFO] Launched tmux session: {session_name}")
                print(f"[INFO] Job directory: {job_dir}")
                try:
                    ls_proc = subprocess.run([tmux_path, 'ls'], capture_output=True, text=True)
                    ls_out = (ls_proc.stdout or '').strip()
                    if not ls_out and ls_proc.stderr:
                        ls_out = ls_proc.stderr.strip()
                    if ls_out:
                        print("\n[INFO] tmux sessions list:\n" + ls_out + "\n")
                except Exception as _e:
                    print(f"[WARN] Unable to list tmux sessions: {_e}")
                print(f"[INFO] To inspect logs: tmux attach -t {session_name}")
                return
            except Exception as e:
                print(f"[WARN] Failed to start tmux session ({e}). Falling back to background process...")

        # fallback: non-blocking background process with stdout/err redirected
        print("[DEBUG] subprocess launch command:", self.python_exec, runner_path.as_posix())
        with open(log_path, 'a') as log_file:
            proc = subprocess.Popen([self.python_exec, runner_path.as_posix()], stdout=log_file, stderr=subprocess.STDOUT,
                                    close_fds=True)
        (job_dir / 'pid').write_text(str(proc.pid))
        self._append_session_log(f"pid_{proc.pid}", job_dir)
        print("[INFO] Started background process PID", proc.pid)
        print(f"[INFO] Logs: {log_path}")
        print(f"[INFO] Job directory: {job_dir}")

    def _enqueue(self, job_dir: Path, runner_path: Path):
        """Mark a job as queued and persist queue metadata."""
        queued_flag = job_dir / "QUEUED"
        queued_flag.write_text(datetime.datetime.now().isoformat() + "Z")
        queue_entry = {
            "timestamp": datetime.datetime.now().isoformat() + "Z",
            "job_id": job_dir.name,
            "job_dir": str(job_dir),
            "runner_path": str(runner_path),
        }
        try:
            with (self.queue_dir / "queue.log").open("a") as f:
                f.write(json.dumps(queue_entry, sort_keys=True) + "\n")
        except Exception:
            pass
        print(f"[INFO] Queued job: {job_dir}")

    def _start_queue_worker(self):
        """Start queue worker; prefer tmux so users can inspect it.

        The worker itself enforces a single active instance via a filesystem lock.
        Here we additionally avoid spawning duplicate tmux sessions.
        """

        worker_cmd = (
            f"{shlex.quote(self.python_exec)} -m eegkit.views.job_queue_worker "
            f"--jobs-root {shlex.quote(str(self.jobs_root))}"
        )

        tmux_path = shutil.which("tmux")
        if tmux_path:
            session_name = "eeg_queue_worker"
            worker_log = self.queue_dir / "worker.log"
            shell_cmd = f"{worker_cmd} 2>&1 | tee -a {shlex.quote(str(worker_log))}"

            # If session already exists, don't start a second one.
            has_proc = subprocess.run(
                [tmux_path, "has-session", "-t", session_name],
                capture_output=True,
                text=True,
            )
            if has_proc.returncode == 0:
                print(f"[INFO] Queue worker tmux session already running: {session_name}")
                print(f"[INFO] Attach: tmux attach -t {session_name}")
                print(f"[INFO] Log: {worker_log}")
                return

            cmd = [
                tmux_path,
                "new-session",
                "-d",
                "-s",
                session_name,
                shell_cmd,
            ]
            try:
                subprocess.run(cmd, check=True)
                print(f"[INFO] Queue worker started in tmux session: {session_name}")
                print(f"[INFO] Attach: tmux attach -t {session_name}")
                print(f"[INFO] Log: {worker_log}")
                return
            except Exception as e:
                print(f"[WARN] Failed to start queue worker in tmux ({e}). Falling back to background process...")

        # Fallback: detached background process (no tmux available)
        try:
            subprocess.Popen(
                [self.python_exec, "-m", "eegkit.views.job_queue_worker", "--jobs-root", str(self.jobs_root)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            print("[INFO] Queue worker triggered (single-worker lock prevents duplicates).")
        except Exception as e:
            print(f"[WARN] Failed to start queue worker ({e}).")

    def _append_session_log(self, session_name: str, job_dir: Path):
        """Record tmux session or process id alongside the job directory."""
        record = {
            "timestamp": datetime.datetime.now().isoformat() + "Z",
            "session": session_name,
            "job_dir": str(job_dir),
            "job_id": job_dir.name,
            "task": job_dir.parent.parent.name,  # task dir
            "plot_type": job_dir.parent.name,  # plot type dir
        }
        try:
            with self._runs_log_path.open('a') as f:
                f.write(json.dumps(record, sort_keys=True) + '\n')
        except Exception:
            pass
