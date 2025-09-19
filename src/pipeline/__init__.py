"""Compatibility shims for the legacy ``src.pipeline`` package.

The project now organises shared helpers under dedicated packages:

* :mod:`src.features.dates`
* :mod:`src.features.engineering`
* :mod:`src.models.nowcast_design`

Imports from :mod:`src.pipeline` remain available for existing scripts by
re-exporting the updated locations here.
"""

from src.features import dates  # re-exported for backwards compatibility
from src.features import engineering as features
from src.models import nowcast_design as nowcast

__all__ = ["dates", "features", "nowcast"]
