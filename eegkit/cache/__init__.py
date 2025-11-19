"""Caching utilities for persisting filtered raws, epochs and evoked data.

Exports:
	LocalCache: File-system cache manager.
	CacheKey: Immutable key describing artifact & processing params.
	PIPELINE_VERSION: Stamp incremented when processing logic changes.
"""

from .cache_service import LocalCache, CacheKey

# Central pipeline version stamp used by cache keys
PIPELINE_VERSION = "v3"

__all__ = ["LocalCache", "CacheKey", "PIPELINE_VERSION"]
