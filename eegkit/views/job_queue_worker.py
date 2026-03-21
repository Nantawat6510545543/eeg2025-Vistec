"""Background worker that executes queued jobs sequentially.

This worker enforces one-job-at-a-time execution to avoid GPU contention.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from eegkit.utils.system import configure_logging

logger = logging.getLogger(__name__)


def _iter_queued_jobs(jobs_root: Path) -> list[Path]:
    """Return queued job directories ordered by queue timestamp."""

    queued_dirs = [p.parent for p in jobs_root.glob("*/*/*/QUEUED")]

    def _queued_ts(job_dir: Path) -> float:
        flag = job_dir / "QUEUED"
        try:
            return flag.stat().st_mtime
        except Exception:
            return float("inf")

    return sorted(queued_dirs, key=_queued_ts)


def _run_one_job(job_dir: Path) -> int:
    """Execute a single queued job synchronously and return exit code."""

    runner_path = job_dir / "run.py"
    log_path = job_dir / "job.log"

    if not runner_path.exists():
        logger.error("Missing runner file for queued job: %s", runner_path)
        (job_dir / "ERROR").write_text("Missing run.py")
        return 1

    queued_flag = job_dir / "QUEUED"
    running_flag = job_dir / "RUNNING"
    done_flag = job_dir / "DONE"

    try:
        if queued_flag.exists():
            queued_flag.unlink()
    except Exception:
        pass

    running_flag.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    cmd = [sys.executable or "python", str(runner_path)]
    logger.info("[QUEUE] Running job %s", job_dir)
    with log_path.open("a") as f:
        f.write("\n[QUEUE] Starting queued job\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    exit_code = int(proc.returncode)
    try:
        running_flag.unlink(missing_ok=True)
    except TypeError:
        if running_flag.exists():
            running_flag.unlink()

    if exit_code == 0:
        done_flag.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        logger.info("[QUEUE] Job succeeded: %s", job_dir)
    else:
        (job_dir / "ERROR").write_text(f"Queued job failed with exit code {exit_code}")
        logger.error("[QUEUE] Job failed (%s): %s", exit_code, job_dir)

    return exit_code


def main() -> int:
    """Acquire queue lock and drain queued jobs."""

    parser = argparse.ArgumentParser(description="Drain EEG job queue sequentially")
    parser.add_argument("--jobs-root", required=True, help="Path to jobs root directory")
    args = parser.parse_args()

    configure_logging()

    jobs_root = Path(args.jobs_root)
    queue_dir = jobs_root / ".queue"
    queue_dir.mkdir(parents=True, exist_ok=True)

    lock_path = queue_dir / "worker.lock"
    with lock_path.open("w") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info("[QUEUE] Worker already active; exiting.")
            return 0

        logger.info("[QUEUE] Worker started for %s", jobs_root)

        while True:
            queued_jobs = _iter_queued_jobs(jobs_root)
            if not queued_jobs:
                break

            for job_dir in queued_jobs:
                try:
                    _run_one_job(job_dir)
                except Exception as exc:
                    logger.exception("[QUEUE] Unexpected error while running %s: %s", job_dir, exc)
                    try:
                        (job_dir / "ERROR").write_text(str(exc))
                    except Exception:
                        pass

        logger.info("[QUEUE] Queue is empty; worker exiting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
