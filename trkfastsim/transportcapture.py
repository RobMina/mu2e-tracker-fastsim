"""Passive algorithm-state observations, never an ordered physical material history."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ("run", "subrun", "event", "sp")
SCHEMA = "trkfastsim.transport-observations.v1"
DISPOSITIONS = ("no_candidates", "final_non_kink", "unresolved_stop", "iteration_cap",
                "not_processed", "no_support", "extent_complete")
STATE = ("path_mm", "position_mm", "direction", "model_ke_MeV", "momentum_MeV_c",
         "lost_support_MeV", "lost_thin_MeV")


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _copy(values):
    return {k: v.copy() for k, v in values.items()}


def source_paths():
    """Implementation dependencies of primary NumPy transport, identity and surface output."""
    from mu2efastsim import crossings as core_crossings, eloss, materials, tables as core_tables, transport, world
    from . import crossings, entryledger, geometry, ledger, steps, surfaces, tables, worldmods
    modules = (core_crossings, core_tables, eloss, materials, transport, world, crossings, entryledger,
               geometry, ledger, steps, surfaces, tables, worldmods)
    return {**{module.__name__: Path(module.__file__) for module in modules}, __name__: Path(__file__)}


def source_hashes():
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_paths().items()}


class PhaseCapture:
    """One algorithm phase with copied observations and an explicit disposition per input lane.

    A recorded entry or kink is an established *model* state, not a physical stopping
    location. Momentum may use the legacy floor and need not equal the raw model KE.
    Path is the producer's coordinate plus a separately recorded phase offset. Neither
    material chronology nor flight-time interpolation is inferred. last_final remains
    a separately recorded computational restart, never silently promoted to terminal.
    """

    def __init__(self, identity, *, path_offset=None, settings=None):
        identity = np.asarray(identity)
        if (identity.ndim != 2 or identity.shape[1] != 4 or identity.dtype.kind not in "iu"
                or (identity < 0).any() or (identity > np.iinfo(np.int64).max).any()):
            raise ValueError("phase requires nonnegative int64-compatible (N,4) identities")
        if len(np.unique(identity, axis=0)) != len(identity):
            raise ValueError("phase identities must be unique")
        self._identity = identity.astype(np.int64, copy=True)
        n = len(identity)
        offset = np.zeros(n) if path_offset is None else np.asarray(path_offset, float)
        if offset.shape != (n,) or not np.isfinite(offset).all():
            raise ValueError("phase path offsets must be finite (N,) coordinates")
        self._offset = offset.copy()
        self.settings = _json(settings or {})
        self._records = []
        self._status = np.full(n, "active", dtype="U24")
        self._iteration = np.full(n, -1, dtype=np.int64)
        self._last = {k: np.full((n, 3) if k in ("position_mm", "direction") else n, np.nan) for k in STATE}
        self._last_kind = np.full(n, "unobserved", dtype="U24")
        self._last_iteration = np.full(n, -1, np.int64)
        self._complete = False

    def _lanes(self, lanes):
        lanes = np.asarray(lanes)
        if lanes.ndim != 1 or lanes.dtype.kind not in "iu" or ((lanes < 0) | (lanes >= len(self._identity))).any():
            raise ValueError("invalid phase lanes")
        return lanes.astype(np.int64, copy=False)

    def record(self, kind, lanes, *, iteration=0, **values):
        """Copy one vectorized operation batch; no sampler, geometry or RNG calls."""
        if self._complete:
            raise ValueError("phase capture is already complete")
        lanes = self._lanes(lanes)
        if not isinstance(kind, str) or not kind or any(k in values for k in ("identity", "lane", "iteration")):
            raise ValueError("invalid record name or reserved column")
        arrays = {"identity": self._identity[lanes].copy(), "lane": lanes.copy(),
                  "iteration": np.full(len(lanes), iteration, np.int64)}
        for name, value in values.items():
            value = np.asarray(value)
            if value.dtype.kind not in "biufU" or value.ndim < 1 or value.shape[0] != len(lanes):
                raise ValueError(f"{name} must be a row-aligned numeric/string array")
            arrays[name] = value.copy()
        self._records.append((kind, arrays))

    def observe(self, kind, lanes, *, path_mm, position_mm, direction, model_ke_MeV,
                momentum_MeV_c, lost_support_MeV, lost_thin_MeV, iteration=0):
        """Record selected entry/kink states and remember the last row of each lane in producer order."""
        lanes = self._lanes(lanes)
        values = dict(path_mm=np.asarray(path_mm) + self._offset[lanes], position_mm=np.asarray(position_mm),
                      direction=np.asarray(direction), model_ke_MeV=np.asarray(model_ke_MeV),
                      momentum_MeV_c=np.asarray(momentum_MeV_c), lost_support_MeV=np.asarray(lost_support_MeV),
                      lost_thin_MeV=np.asarray(lost_thin_MeV))
        if (self._status[lanes] != "active").any():
            raise ValueError("cannot observe a terminated phase lane")
        for k, value in values.items():
            if value.shape != ((len(lanes), 3) if k in ("position_mm", "direction") else (len(lanes),)):
                raise ValueError(f"invalid observation shape for {k}")
        self.record(kind, lanes, iteration=iteration, **values)
        latest = np.full(len(self._identity), -1, np.int64)
        np.maximum.at(latest, lanes, np.arange(len(lanes)))
        used = np.flatnonzero(latest >= 0)
        for k, value in values.items():
            self._last[k][used] = value[latest[used]]
        self._last_kind[used] = kind
        self._last_iteration[used] = iteration

    def terminate(self, lanes, disposition, *, iteration=0):
        lanes = self._lanes(lanes)
        if self._complete or disposition not in DISPOSITIONS or len(np.unique(lanes)) != len(lanes):
            raise ValueError("invalid phase termination")
        if (self._status[lanes] != "active").any():
            raise ValueError("phase lane already terminated")
        self._status[lanes] = disposition
        self._iteration[lanes] = iteration

    def finish(self, **legacy_restart):
        """Seal all dispositions and retain exact legacy restart arrays as a separate record."""
        if (self._status == "active").any():
            raise ValueError("capture has unclassified phase lanes")
        self.record("legacy_restart", np.arange(len(self._identity)), **legacy_restart)
        self._complete = True

    @property
    def complete(self):
        return self._complete

    def records(self, kind=None):
        return [(name, _copy(values)) for name, values in self._records if kind is None or name == kind]

    def summary(self):
        """Per input lane, disposition and last observed model state; never an extrapolated endpoint."""
        result = {k: self._identity[:, j].copy() for j, k in enumerate(KEY)}
        result.update(disposition=self._status.copy(), disposition_iteration=self._iteration.copy(),
                      last_kind=self._last_kind.copy(), last_iteration=self._last_iteration.copy(),
                      phase_path_offset_mm=self._offset.copy())
        for name, value in self._last.items():
            if value.ndim == 2:
                result.update({f"last_{name}_{axis}": value[:, j].copy() for j, axis in enumerate("xyz")})
            else:
                result[f"last_{name}"] = value.copy()
        return pd.DataFrame(result)


class TransportCapture:
    """Per-call primary-transport observation sidecar, created only by explicit opt-in.

    Settings and source hashes describe the loaded implementation, not verified input
    provenance. The caller must bind actual generator/geometry contents when publishing.
    No phase is joined into a physical trajectory; exit can start from a stale restart.
    """

    def __init__(self, settings):
        import platform
        self.settings = _json(settings)
        self.sources = source_hashes()
        self.environment = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                system=platform.system(), machine=platform.machine())
        self._phases = {}
        self._handoff = None
        self.complete = False

    def bind(self, mu, **handoff):
        if self._handoff is not None:
            raise ValueError("transport capture cannot be rebound")
        columns = []
        for key in ("run", "subrun", "event", "mukey"):
            value = mu[key]
            if (not pd.api.types.is_integer_dtype(value.dtype) or value.isna().any()
                    or (value < 0).any() or (value > np.iinfo(np.int64).max).any()):
                raise ValueError("handoff identities must be nonnegative int64-compatible integers")
            columns.append(value.to_numpy(dtype=np.int64))
        identity = np.column_stack(columns)
        PhaseCapture(identity)
        arrays = {"identity": identity.astype(np.int64, copy=True)}
        for name, value in handoff.items():
            value = np.asarray(value)
            if value.ndim < 1 or value.shape[0] != len(mu):
                raise ValueError("handoff values must align with all muons")
            arrays[name] = value.copy()
        self._handoff = arrays

    @property
    def handoff(self):
        return None if self._handoff is None else _copy(self._handoff)

    def phase(self, name, lanes, *, path_offset=None, settings=None):
        if self.complete or name in self._phases or self._handoff is None:
            raise ValueError("phase creation is not allowed")
        result = PhaseCapture(self._handoff["identity"][lanes], path_offset=path_offset, settings=settings)
        self._phases[name] = result
        return result

    def get_phase(self, name):
        return self._phases[name]

    def finish(self):
        if self.complete or self._handoff is None or not all(p.complete for p in self._phases.values()):
            raise ValueError("capture is incomplete or already finished")
        self.complete = True

    def save(self, path):
        """Save complete per-call arrays without pickle or overwrite; not an atomic campaign publisher."""
        if not self.complete:
            raise ValueError("cannot save incomplete transport capture")
        arrays, phases = {}, {}
        for name, value in self._handoff.items():
            arrays[f"handoff_{name}"] = value
        for name, phase in self._phases.items():
            records = []
            for i, (kind, values) in enumerate(phase.records()):
                record = {"kind": kind, "columns": {}}
                for key, value in values.items():
                    array = f"{name}_{i}_{key}"
                    arrays[array] = value
                    record["columns"][key] = array
                records.append(record)
            for key, value in phase.summary().items():
                arrays[f"{name}_summary_{key}"] = value.to_numpy(dtype=str if pd.api.types.is_string_dtype(value.dtype) else None)
            phases[name] = {"settings": phase.settings, "records": records}
        metadata = dict(schema=SCHEMA, scope="algorithm observations; not physical material history",
                        settings=self.settings, source_sha256=self.sources, environment=self.environment, phases=phases)
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, manifest_json=np.array(json.dumps(metadata, allow_nan=False)), **arrays)
