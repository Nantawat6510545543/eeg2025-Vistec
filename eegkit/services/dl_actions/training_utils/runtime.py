"""Runtime helpers for DL actions (device selection, seeding).

Why this exists:
- Some NAS/FUSE environments and some GPU stacks can fail hard in native code.
- For RTX 50xx (e.g. sm_120) with older PyTorch builds, cuDNN-backed conv ops may
  crash even when basic CUDA ops work.

This module centralizes device selection and applies a safe workaround:
- If CUDA is selected and the GPU capability tag (e.g. sm_120) is not in
  torch.cuda.get_arch_list(), disable cuDNN so conv ops use non-cuDNN kernels.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _normalize_device_value(device_value: Optional[str]) -> str:
	value = str(device_value or "auto").strip().lower()
	if value in {"", "none"}:
		return "auto"
	return value


def _cuda_capability_tag(device_index: int = 0) -> str:
	try:
		import torch

		major, minor = torch.cuda.get_device_capability(device_index)
		return f"sm_{int(major)}{int(minor)}"
	except Exception:
		return "sm_unknown"


def _torch_arch_list() -> list[str]:
	try:
		import torch

		return list(torch.cuda.get_arch_list())
	except Exception:
		return []


def _set_global_seeds(seed: Optional[int]) -> None:
	if seed is None:
		return

	random.seed(seed)
	np.random.seed(seed)

	try:
		import torch

		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
	except Exception:
		# If torch isn't available or CUDA isn't healthy, still keep python/numpy seeded.
		return


def _maybe_disable_cudnn_for_unsupported_cuda_arch(device: "torch.device") -> None:
	import torch

	force_disable = str(os.getenv("EEGKIT_DISABLE_CUDNN", "")).strip().lower() in {"1", "true", "yes"}
	if force_disable:
		if torch.backends.cudnn.enabled:
			logger.warning("EEGKIT_DISABLE_CUDNN is set; disabling cuDNN")
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.benchmark = False
		return

	if device.type != "cuda":
		return

	if not torch.backends.cudnn.enabled:
		return

	sm_tag = _cuda_capability_tag(0)
	arch_list = _torch_arch_list()

	# If torch provides an arch list and the current GPU isn't included, this
	# environment is very likely to be unstable for cuDNN kernels.
	if arch_list and sm_tag not in arch_list:
		logger.warning(
			"CUDA device capability %s is not in this PyTorch build arch list (%s). "
			"Disabling cuDNN to avoid runtime loader/kernel crashes.",
			sm_tag,
			",".join(arch_list),
		)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.benchmark = False


def resolve_device_and_seed(device_value: Optional[str], seed: Optional[int] = None) -> Tuple[str, "torch.device"]:
	"""Resolve a training device and apply global RNG seeding.

	Args:
		device_value: One of {"auto", "cpu", "cuda"} (or "cuda:0" etc.).
		seed: Global seed value.

	Returns:
		(device_str, device)
	"""

	import torch

	device_str = _normalize_device_value(device_value)
	_set_global_seeds(seed)

	if device_str in {"auto", "cuda"}:
		resolved = "cuda" if torch.cuda.is_available() else "cpu"
	elif device_str.startswith("cuda"):
		resolved = device_str if torch.cuda.is_available() else "cpu"
	else:
		resolved = "cpu"

	device = torch.device(resolved)
	_maybe_disable_cudnn_for_unsupported_cuda_arch(device)
	return resolved, device
