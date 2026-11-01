"""Alias of trkfastsim.kit.kit_task (moved there; this alias is kept for one release, for importers in the other repositories)."""
import sys

from .kit import kit_task as _moved

sys.modules[__name__] = _moved
