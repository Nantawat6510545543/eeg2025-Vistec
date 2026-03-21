"""AI model collection.

Auto-discovered by EEGAIService based on nn.Module subclasses defined here.
Keep imports lightweight; avoid side effects. Models should follow a simple
constructor contract so the service can introspect them (e.g., accept
in_channels, n_classes or num_classes).
"""

from .CNNLSTMDense import CNNLSTMDense
from .EEGNetAliasVishnu import EEGNetAliasVishnu
from .EEGNetBinary import EEGNetBinary
from .EEGNetReg import EEGNetReg
from .EEGNetMultiOutput import EEGNetMultiOutput
from .EEGNetMultiReg import EEGNetMultiReg
from .simpleNN import SimpleNN

__all__ = [
    "EEGNet",
    "EEGNetBinary",
    "EEGNetMultiOutput",
    "SimpleNN",
    "CNNLSTMDense",
    "EEGNetMultiReg",
    "EEGNetReg",
]
