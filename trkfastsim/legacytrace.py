"""Passive legacy operation trace; aggregate material columns are not physical segment histories."""
import hashlib
import json
from pathlib import Path

import numpy as np

from mu2efastsim import transport
from .transportledger import _json_copy, _validate_manifest, medium


class LegacyTrace:
    """Per-call array snapshots with full muon identities and explicit column-coordinate scope.

    This diagnostic records algorithm states, including invalid physical states, without repairing them.
    A new collector is required for each make_steps call. It is not a SegmentHistory producer.
    """

    def __init__(self, metadata):
        _validate_manifest(metadata)
        self.metadata = _json_copy(metadata)
        self.records = []
        self._keys = None
        self.effective_settings = None

    def configure(self, settings):
        if self.effective_settings is not None or self._keys is not None:
            raise ValueError("a transport trace cannot be reused across make_steps calls")
        from . import steps
        self.effective_settings = _json_copy(settings)
        self.effective_settings["source_sha256"] = {
            name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in (("tracker_steps", steps), ("core_transport", transport))}

    def bind(self, mu):
        if self._keys is not None:
            raise ValueError("a transport trace cannot be reused across make_steps calls")
        self._keys = mu[["run", "subrun", "event", "mukey"]].to_numpy(dtype=np.int64, copy=True)
        return TraceScope(self, np.arange(len(mu)))

    def _append(self, stage, kind, **values):
        record = {"stage": stage, "kind": kind}
        record.update({k: v.copy() if isinstance(v, np.ndarray) else _json_copy(v) for k, v in values.items()})
        self.records.append(record)

    def save(self, path):
        """Write heterogeneous operation batches as arrays plus JSON, without pickle or overwrite."""
        arrays, records = {}, []
        for i, record in enumerate(self.records):
            item = {}
            for name, value in record.items():
                if isinstance(value, np.ndarray):
                    key = f"batch_{i}_{name}"
                    arrays[key] = value
                    item[name] = {"array": key}
                else:
                    item[name] = value
            records.append(item)
        arrays["identities"] = self._keys if self._keys is not None else np.empty((0, 4), dtype=np.int64)
        manifest = dict(schema="trkfastsim.legacy-operation-trace.v1", provenance=self.metadata,
                        effective_settings=self.effective_settings,
                        scope="algorithm operations; material-column distance is not physical path order",
                        identities={"array": "identities"}, records=records)
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, manifest_json=np.asarray(json.dumps(manifest, allow_nan=False)), **arrays)


class TraceScope:
    """Map local transporter lanes back to the collector's full primary-muon identities."""

    def __init__(self, owner, lanes):
        self.owner, self.lanes = owner, np.asarray(lanes, dtype=np.int64)

    def subset(self, lanes):
        return TraceScope(self.owner, self.lanes[lanes])

    def emit(self, stage, kind, lanes, **values):
        lanes = np.asarray(lanes, dtype=np.int64)
        self.owner._append(stage, kind, identity=self.owner._keys[self.lanes[lanes]], **values)

    def slab_loss(self, rng, mat, p, length, lanes, *, stage, **context):
        """Run the existing core sampler once and capture its column-space operations unchanged."""
        call = len(self.owner.records)
        self.emit(stage, "slab_begin", lanes, call=call, material=medium(mat, cut_MeV=1e9, loss_basis="unrestricted"),
                  p_sampler_MeV_c=np.array(p, copy=True), material_column_mm=np.array(length, copy=True), **context)
        result = transport.slab_loss(rng, mat, p, length,
                                    trace=lambda batch: self.owner._append(stage, "slab_core", call=call, **batch))
        self.emit(stage, "slab_return", lanes, call=call, loss_MeV=result)
        return result
