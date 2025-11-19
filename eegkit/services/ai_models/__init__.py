"""AI model collection.

Auto-discovered by EEGAIService based on nn.Module subclasses defined here.
Keep imports lightweight; avoid side effects. Models should follow a simple
constructor contract so the service can introspect them (e.g., accept
in_channels, n_classes or num_classes).
"""

from .CNNLSTMDense import CNNLSTMDense
from .EEGNet import EEGNet
from .EEGNetMultiOutput import EEGNetMultiOutput
from .EEGNetMultiReg import EEGNetMultiReg
from .simpleNN import SimpleNN

__all__ = [
    "EEGNet",
    "EEGNetMultiOutput",
    "SimpleNN",
    "CNNLSTMDense",
    "EEGNetMultiReg",
]
