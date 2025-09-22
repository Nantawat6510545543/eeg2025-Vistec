from dataclasses import dataclass
from pathlib import Path
import hashlib, json, os
import mne


def _repo_root(start: Path) -> Path:
    for p in [*start.parents, start]:
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
    return start


def _hash_of_dict(d):
    s = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class CacheKey:
    subject: str
    task: str
    run: str | None
    stage: str  # "rawfilt" or "epochs"
    params: dict  # DTO -> dict
    pipeline_ver: str  # bump when processing logic changes

    def subdir(self):
        r = f"run-{self.run}" if self.run else "run-none"
        return f"{self.subject}/{self.task}/{r}/{self.stage}"

    def filename_stem(self):
        return f"{_hash_of_dict(self.params)}-{self.pipeline_ver}"


class LocalCache:
    def __init__(self, base_dir: Path | None = None, pipeline_ver: str = "v1"):
        self.repo_root = _repo_root(Path.cwd())
        self.base = base_dir or (self.repo_root / ".eegcache")
        self.base.mkdir(exist_ok=True)
        self.pipeline_ver = pipeline_ver

    def _path_for(self, key: CacheKey, type, ext: str) -> Path:
        d = self.base / key.subdir()
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key.filename_stem()}_{type}.{ext}"

    def load_raw_filtered(self, key: CacheKey):
        p = self._path_for(key, "eeg", "fif")
        if p.exists():
            return mne.io.read_raw_fif(p.as_posix(), preload=False, verbose="ERROR")
        return None

    def save_raw_filtered(self, raw, key: CacheKey):
        p = self._path_for(key, "eeg", "fif")
        raw.save(p.as_posix(), overwrite=True)
        return p

    def load_epochs(self, key: CacheKey):
        p = self._path_for(key, "epo", "fif")
        if not p.exists():
            return None, None
        epochs = mne.read_epochs(p.as_posix(), preload=False, verbose="ERROR")
        labels_file = p.with_suffix(".labels.json")
        labels = None
        if labels_file.exists():
            with open(labels_file, "r") as f:
                labels = json.load(f)
        return epochs, labels

    def save_epochs(self, epochs, key: CacheKey, labels=None):
        p = self._path_for(key, "epo", "fif")
        epochs.save(p.as_posix(), overwrite=True)
        if labels is not None:
            labels_file = p.with_suffix(".labels.json")
            with open(labels_file, "w") as f:
                try:
                    from numpy import ndarray
                    if isinstance(labels, ndarray):
                        json.dump(labels.tolist(), f)
                    elif isinstance(labels, (list, tuple)):
                        json.dump(list(labels), f)
                    elif isinstance(labels, dict):
                        json.dump(labels, f)
                    else:
                        json.dump(labels, f)
                except Exception:
                    json.dump(str(labels), f)
        return p
