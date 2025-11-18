"""Pipeline package.

Task-level EEG processing components:
- task_model: orchestrates filtering, epoching, and evoked caching per task
- task_loader: data loading from disk and task discovery
- task_processor: epoch building per task via registered preprocessors
- constants: shared arrays/mappings used by preprocessors
"""

from . import task_model as task_model  # noqa: F401
from . import task_loader as task_loader  # noqa: F401
from . import task_processor as task_processor  # noqa: F401
from . import constants as constants  # noqa: F401

__all__ = [
    'task_model',
    'task_loader',
    'task_processor',
    'constants',
]
