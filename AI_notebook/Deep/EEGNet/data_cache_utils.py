"""Load and cache EEG notebook arrays using a stable key from params."""

from dataclasses import asdict, is_dataclass
from pathlib import Path
import hashlib
import json
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _json_default(obj: Any):
    """Convert non-JSON values into stable string-compatible values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def params_to_dict(params_obj: Any) -> Dict[str, Any]:
    """Convert params object to a dictionary for stable hashing."""
    if isinstance(params_obj, dict):
        return dict(params_obj)
    if is_dataclass(params_obj):
        return asdict(params_obj)
    if hasattr(params_obj, "__dict__"):
        return dict(params_obj.__dict__)
    return {"params_repr": str(params_obj)}


def build_cache_key(params_obj: Any, n_subjects: int, key_prefix: str = "ccd_eegnet") -> str:
    """Build cache key from params content and subject count."""
    params_dict = params_to_dict(params_obj)
    params_json = json.dumps(params_dict, sort_keys=True, default=_json_default, separators=(",", ":"))
    params_hash = hashlib.sha256(params_json.encode("utf-8")).hexdigest()[:12]
    return f"{key_prefix}_n{int(n_subjects)}_p{params_hash}"


def get_cache_paths(cache_dir: Path, cache_key: str) -> Dict[str, Path]:
    """Return standardized cache file paths for a cache key."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "X": cache_dir / f"X_{cache_key}.npy",
        "y": cache_dir / f"y_{cache_key}.npy",
        "groups": cache_dir / f"groups_{cache_key}.npy",
        "responds": cache_dir / f"responds_{cache_key}.npy",
        "meta": cache_dir / f"meta_{cache_key}.json",
    }


def save_cache(
    cache_dir: Path,
    cache_key: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    responds: Optional[np.ndarray] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Save arrays to cache and return file paths."""
    paths = get_cache_paths(cache_dir, cache_key)

    np.save(paths["X"], np.asarray(X))
    np.save(paths["y"], np.asarray(y))
    np.save(paths["groups"], np.asarray(groups).astype(str))

    if responds is not None:
        np.save(paths["responds"], np.asarray(responds))

    meta = {
        "cache_key": cache_key,
        "X_shape": list(np.asarray(X).shape),
        "y_shape": list(np.asarray(y).shape),
        "groups_shape": list(np.asarray(groups).shape),
        "has_responds": responds is not None,
    }
    if extra_meta:
        meta.update(extra_meta)

    paths["meta"].write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return paths


def load_cache(
    cache_dir: Path,
    cache_key: str,
    require_responds: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict[str, Path]]:
    """Load arrays from cache; return None arrays if required files are missing."""
    paths = get_cache_paths(cache_dir, cache_key)

    required = [paths["X"], paths["y"], paths["groups"]]
    if require_responds:
        required.append(paths["responds"])

    if not all(path.exists() for path in required):
        return None, None, None, None, paths

    X = np.load(paths["X"])
    y = np.load(paths["y"])
    groups = np.load(paths["groups"])
    responds = np.load(paths["responds"]) if paths["responds"].exists() else None
    return X, y, groups, responds, paths


def load_or_cache(
    cache_dir: Path,
    params_obj: Any,
    n_subjects: int,
    build_fn,
    key_prefix: str = "ccd_eegnet",
    require_responds: bool = False,
):
    """Load cached arrays by params key, or build and cache them if missing.

    `build_fn` must return a tuple: (X, y, groups, responds_or_none).
    """
    cache_key = build_cache_key(params_obj=params_obj, n_subjects=n_subjects, key_prefix=key_prefix)
    X, y, groups, responds, paths = load_cache(
        cache_dir=cache_dir,
        cache_key=cache_key,
        require_responds=require_responds,
    )

    if X is not None:
        return {
            "status": "loaded",
            "cache_key": cache_key,
            "X": X,
            "y": y,
            "groups": groups,
            "responds": responds,
            "paths": paths,
        }

    X, y, groups, responds = build_fn()
    save_cache(
        cache_dir=cache_dir,
        cache_key=cache_key,
        X=X,
        y=y,
        groups=groups,
        responds=responds,
    )

    return {
        "status": "built_and_cached",
        "cache_key": cache_key,
        "X": np.asarray(X),
        "y": np.asarray(y),
        "groups": np.asarray(groups),
        "responds": None if responds is None else np.asarray(responds),
        "paths": paths,
    }
