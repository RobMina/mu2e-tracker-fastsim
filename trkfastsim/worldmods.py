"""Alias of trkfastsim.sim.worldmods (moved there; this alias is kept for one release, for importers in the other repositories)."""
import sys

from .sim import worldmods as _moved

sys.modules[__name__] = _moved
