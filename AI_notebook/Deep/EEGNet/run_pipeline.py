"""Run EEGNet data build and training with a short command interface."""

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


def parse_args():
    """Parse command-line options for pipeline run."""
    parser = argparse.ArgumentParser(description="Build cache and train EEGNet with a simple command.")
    parser.add_argument("--data_dir", type=str, default="/mount/NAS-public-dataset/HBN-EEG")
    parser.add_argument("--n_subjects", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--run_name", type=str, default="eegnet_run")
    parser.add_argument("--cuda_device", type=str, default="cpu")
    parser.add_argument("--cache_dir", type=str, default="/mount/NAS-workspace-portal/eeg2025-Vistec/models/data")
    parser.add_argument("--output_root", type=str, default="/mount/NAS-workspace-portal/eeg2025-Vistec/jobs")
    parser.add_argument("--channels", type=str, default="69-76,81-83,88,89")
    parser.add_argument("--tmin", type=float, default=-2.0)
    parser.add_argument("--tmax", type=float, default=0.0)

    parser.add_argument("--tmux_session", type=str, default="")
    parser.add_argument("--log_file", type=str, default="")
    parser.add_argument("--run_direct", action="store_true")
    return parser.parse_args()


def run_cmd(cmd):
    """Run a subprocess command and stream output."""
    print("RUN:", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True)


def build_cache_key_from_params(params_json_path: Path, n_subjects: int) -> str:
    """Compute cache key from saved params file and subject count."""
    module_path = SCRIPT_DIR / "data_cache_utils.py"
    spec = importlib.util.spec_from_file_location("data_cache_utils", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load data_cache_utils from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_cache_key = module.build_cache_key

    params_obj = json.loads(params_json_path.read_text(encoding="utf-8"))
    return build_cache_key(params_obj=params_obj, n_subjects=n_subjects, key_prefix="ccd_eegnet")


def run_pipeline(args):
    """Run build_data then train_classify with computed cache key."""
    params_json = Path(args.cache_dir) / f"params_{args.n_subjects}.json"

    build_cmd = [
        sys.executable,
        "-u",
        str(SCRIPT_DIR / "build_data.py"),
        "--data_dir",
        args.data_dir,
        "--n_subjects",
        str(args.n_subjects),
        "--channels",
        args.channels,
        "--tmin",
        str(args.tmin),
        "--tmax",
        str(args.tmax),
        "--cache_dir",
        args.cache_dir,
        "--params_json_out",
        str(params_json),
    ]
    run_cmd(build_cmd)

    cache_key = build_cache_key_from_params(params_json, args.n_subjects)
    print(f"CACHE_KEY={cache_key}")

    train_cmd = [
        sys.executable,
        "-u",
        str(SCRIPT_DIR / "train_classify.py"),
        "--n_subjects",
        str(args.n_subjects),
        "--cache_key",
        cache_key,
        "--epochs",
        str(args.epochs),
        "--run_name",
        args.run_name,
        "--cuda_device",
        args.cuda_device,
        "--cache_dir",
        args.cache_dir,
        "--output_root",
        args.output_root,
    ]
    run_cmd(train_cmd)


def start_tmux(args):
    """Start detached tmux session for the pipeline."""
    log_file = args.log_file or str(Path(args.output_root) / "logs" / f"{args.tmux_session}.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(SCRIPT_DIR / "run_pipeline.py"),
        "--data_dir",
        args.data_dir,
        "--n_subjects",
        str(args.n_subjects),
        "--epochs",
        str(args.epochs),
        "--run_name",
        args.run_name,
        "--cuda_device",
        args.cuda_device,
        "--cache_dir",
        args.cache_dir,
        "--output_root",
        args.output_root,
        "--channels",
        args.channels,
        "--tmin",
        str(args.tmin),
        "--tmax",
        str(args.tmax),
        "--run_direct",
    ]
    inner = " ".join(shlex.quote(part) for part in cmd)
    tmux_cmd = (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        f"{inner} 2>&1 | tee {shlex.quote(log_file)}; bash"
    )

    subprocess.run(["tmux", "kill-session", "-t", args.tmux_session], check=False)
    subprocess.run(["tmux", "new-session", "-d", "-s", args.tmux_session, tmux_cmd], check=True)
    print(f"Started tmux session: {args.tmux_session}")
    print(f"Log file: {log_file}")


def main():
    """Entry point for pipeline launcher."""
    args = parse_args()

    if args.tmux_session and not args.run_direct:
        start_tmux(args)
        return

    run_pipeline(args)


if __name__ == "__main__":
    main()
