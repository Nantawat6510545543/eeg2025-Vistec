"""EEG toolkit root package.

Provides the main building blocks of the EEG Workbench:
- controller: coordination layer that wires models and services
- views: notebook UI and job execution helpers
- models: data access, DTOs, and processing pipeline
- services: plotting, grid visualization, data views, and AI helpers
- utils: shared helpers grouped by concern
"""

from . import controller as controller  # noqa: F401
from . import views as views  # noqa: F401
from . import models as models  # noqa: F401
from . import services as services  # noqa: F401
from . import utils as utils  # noqa: F401

__all__ = [
    'controller',
    'views',
    'models',
    'services',
    'utils',
]
