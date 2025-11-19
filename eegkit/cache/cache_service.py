"""File-system backed cache for intermediate EEG processing artifacts.

Provides simple hashed-path storage for:
    - filtered Raw FIF files
    - Epochs FIF + optional labels JSON
    - Evoked FIF files

Corruption detection quarantines bad files automatically.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import mne

log = logging.getLogger(__name__)


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
    """Immutable identifier for a cached artifact.

    Attributes map to subject/task/run/stage and a params dict whose hash
    plus pipeline version form the filename stem.
    """

    subject: str
    task: str
    run: str | None
    stage: str  # "rawfilt" or "epochs"
    params: dict  # DTO -> dict
    pipeline_ver: str  # bump when processing logic changes

    def subdir(self):
        """Return relative subdirectory path for this key."""
        r = f"run-{self.run}" if self.run else "run-none"
        return f"{self.subject}/{self.task}/{r}/{self.stage}"

    def filename_stem(self):
        """Return deterministic stem (params hash + pipeline version)."""
        return f"{_hash_of_dict(self.params)}-{self.pipeline_ver}"


class LocalCache:
    """Persist and retrieve pipeline artifacts (filtered raw, epochs, evoked).

    Uses hashed filenames under .eegcache/<subject>/<task>/run-*/<stage>.
    Corrupted files are renamed with .bad suffix and treated as misses.
    """

    def __init__(self, base_dir: Path | None = None, pipeline_ver: str = "v1"):
        """Initialize cache root directory and record pipeline version stamp."""
        self.repo_root = _repo_root(Path.cwd())
        self.base = base_dir or (self.repo_root / ".eegcache")
        self.base.mkdir(exist_ok=True)
        self.pipeline_ver = pipeline_ver
        log.info("[cache] init base=%s pipeline=%s repo_root=%s", self.base, self.pipeline_ver, self.repo_root)

    # --- logging helpers ---
    def _key_summary(self, key: CacheKey) -> str:
        """Return brief string identifying a cache key for logging."""
        try:
            return (
                f"subject={key.subject}, task={key.task}, run={key.run}, "
                f"stage={key.stage}, ver={key.pipeline_ver}, params_hash={_hash_of_dict(key.params)}"
            )
        except Exception:
            return f"subject={getattr(key, 'subject', None)}, task={getattr(key, 'task', None)}, run={getattr(key, 'run', None)}"

    def _log_get(self, artifact: str, path: Path, key: CacheKey):
        """Log cache GET attempt."""
        log.info("[cache] GET %s file=%s key={%s}", artifact, path.name, self._key_summary(key))

    def _log_hit(self, artifact: str, path: Path, key: CacheKey):
        """Log cache HIT event."""
        log.info("[cache] HIT %s file=%s (subdir=%s, ver=%s)", artifact, path.name, key.subdir(), key.pipeline_ver)

    def _log_miss(self, artifact: str, path: Path, key: CacheKey):
        """Log cache MISS event."""
        log.info("[cache] MISS %s file=%s (subdir=%s, ver=%s)", artifact, path.name, key.subdir(), key.pipeline_ver)

    def _log_save(self, artifact: str, path: Path, key: CacheKey, extra: str | None = None):
        """Log cache SAVE event with optional extra details."""
        if extra:
            log.info("[cache] SAVE %s file=%s (%s)", artifact, path.name, extra)
        else:
            log.info("[cache] SAVE %s file=%s (ver=%s)", artifact, path.name, key.pipeline_ver)

    def _path_for(self, key: CacheKey, type, ext: str) -> Path:
        """Return full path for artifact type+ext ensuring subdir exists."""
        d = self.base / key.subdir()
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{key.filename_stem()}_{type}.{ext}"
        return p

    def load_raw_filtered(self, key: CacheKey):
        """Load filtered Raw if present else return None (quarantine corrupt)."""
        p = self._path_for(key, "eeg", "fif")
        self._log_get("rawfilt", p, key)
        if p.exists():
            self._log_hit("rawfilt", p, key)
            try:
                return mne.io.read_raw_fif(p.as_posix(), preload=False, verbose="ERROR")
            except Exception as e:
                try:
                    bad = p.with_name(p.name + ".bad")
                    p.rename(bad)
                    log.warning("[cache] Corrupt RAWFILT, quarantined as %s (err=%s)", bad.name, e)
                except Exception:
                    log.warning("[cache] Corrupt RAWFILT read (and could not quarantine): %s", e)
                return None
        self._log_miss("rawfilt", p, key)
        return None

    def save_raw_filtered(self, raw, key: CacheKey):
        """Persist filtered Raw to FIF; return path."""
        p = self._path_for(key, "eeg", "fif")
        self._log_save(
            "rawfilt",
            p,
            key,
            extra=f"sfreq={float(raw.info.get('sfreq', 0.0)):.3f}, ch={len(raw.ch_names)}, ver={key.pipeline_ver}",
        )
        raw.save(p.as_posix(), overwrite=True)
        return p

    def load_epochs(self, key: CacheKey):
        """Load epochs + optional labels list; return (epochs, labels) or (None, None)."""
        p = self._path_for(key, "epo", "fif")
        self._log_get("epochs", p, key)
        if not p.exists():
            self._log_miss("epochs", p, key)
            return None, None
        self._log_hit("epochs", p, key)
        try:
            epochs = mne.read_epochs(p.as_posix(), preload=True, verbose="ERROR")
            labels_file = p.with_suffix(".labels.json")
            labels = None
            if labels_file.exists():
                with open(labels_file, "r") as f:
                    labels = json.load(f)
            return epochs, labels
        except Exception as e:
            # Quarantine bad cache and treat as miss so upstream will rebuild
            try:
                bad = p.with_name(p.name + ".bad")
                p.rename(bad)
                labels_file = p.with_suffix(".labels.json")
                if labels_file.exists():
                    try:
                        labels_bad = labels_file.with_name(labels_file.name + ".bad")
                        labels_file.rename(labels_bad)
                    except Exception:
                        labels_file.unlink(missing_ok=True)
                log.warning("[cache] Corrupt EPOCHS, quarantined as %s (err=%s)", bad.name, e)
            except Exception:
                log.warning("[cache] Corrupt EPOCHS read (and could not quarantine): %s", e)
            return None, None

    def save_epochs(self, epochs, key: CacheKey, labels=None):
        """Persist epochs FIF + labels JSON (if provided); return path."""
        p = self._path_for(key, "epo", "fif")
        try:
            n = int(len(epochs))
        except Exception:
            n = -1
        n_str = str(n) if n >= 0 else "?"
        self._log_save("epochs", p, key, extra=f"n={n_str}, ver={key.pipeline_ver}")
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
            log.info("[cache] SAVE epochs labels %s", labels_file)
        return p

    def load_evoked(self, key: CacheKey):
        """Load evoked response if available else None."""
        p = self._path_for(key, "ave", "fif")
        self._log_get("evoked", p, key)
        if not p.exists():
            self._log_miss("evoked", p, key)
            return None
        try:
            self._log_hit("evoked", p, key)
            evk_list = mne.read_evokeds(p.as_posix(), condition=0, verbose="ERROR")
            return evk_list
        except Exception:
            try:
                evk_list = mne.read_evokeds(p.as_posix(), verbose="ERROR")
                return evk_list[0] if evk_list else None
            except Exception:
                return None

    def save_evoked(self, evoked, key: CacheKey):
        """Persist evoked response FIF; return path."""
        p = self._path_for(key, "ave", "fif")
        self._log_save("evoked", p, key)
        evoked.save(p.as_posix(), overwrite=True)
        return p
