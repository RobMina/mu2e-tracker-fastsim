"""Passive, vectorized primary-muon segment history; no sampling or transport policy."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from mu2efastsim.materials import ELEMENTS, Material

SCHEMA = "trkfastsim.transportledger.v1"
KEY = ("run", "subrun", "event", "sp", "segment")
STATUSES = ("known", "unknown", "unmodeled")
DOMAIN = "primary kinetic energy transferred within each physical segment"
UNITS = {"position": "mm", "path": "mm", "energy": "MeV", "time": "ns"}
INTEGER = KEY + ("medium",)
TEXT = ("kind", "terminal", "sample_status", "deposit_status", "escape_status", "dt_status")
REAL = ("path_mm", "ke_in_MeV", "ke_out_MeV", "sampled_loss_MeV", "applied_loss_MeV",
        "deposited_MeV", "escaping_MeV", "dt_ns")
VECTOR = ("entry_mm", "exit_mm")
FIELDS = INTEGER + TEXT + REAL + VECTOR


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _hash(value, size):
    return isinstance(value, str) and len(value) == size and all(c in "0123456789abcdef" for c in value)


def medium(material=None, *, cut_MeV=None, loss_basis="none"):
    """Describe a core Material and its effective cut, or exact modeled vacuum (None)."""
    if material is None:
        if cut_MeV is not None or loss_basis != "none":
            raise ValueError("vacuum has no cut or loss model")
        return {"material": None, "cut_MeV": None, "loss_basis": "none"}
    if (loss_basis not in ("restricted", "unrestricted") or not isinstance(cut_MeV, (int, float))
            or not np.isfinite(cut_MeV) or cut_MeV <= 0):
        raise ValueError("material needs a positive effective cut and explicit loss basis")
    return _json_copy({"material": asdict(material), "cut_MeV": float(cut_MeV), "loss_basis": loss_basis})


def provenance(*, geometry_sha256, tracker_commit, core_commit, inputs, media, frame, settings):
    """Build a JSON manifest from immutable input identities and core material descriptions."""
    result = _json_copy(dict(schema=SCHEMA, units=UNITS, accounting_domain=DOMAIN,
                             geometry_sha256=geometry_sha256, tracker_commit=tracker_commit,
                             core_commit=core_commit, inputs=inputs, media=media, frame=frame,
                             settings=settings, materials_cuts_sha256=_digest(media)))
    _validate_manifest(result)
    return result


def _validate_manifest(meta):
    if meta.get("schema") != SCHEMA or meta.get("units") != UNITS or meta.get("accounting_domain") != DOMAIN:
        raise ValueError("unsupported schema, units or accounting domain")
    for name, size in (("geometry_sha256", 64), ("tracker_commit", 40), ("core_commit", 40)):
        if not _hash(meta.get(name), size):
            raise ValueError(f"{name} must be a full lowercase hash")
    if not isinstance(meta.get("frame"), str) or not meta["frame"].strip():
        raise ValueError("coordinate frame is required")
    if not isinstance(meta.get("settings"), dict) or not isinstance(meta.get("inputs"), list):
        raise ValueError("settings and ordered input manifest are required")
    for source in meta["inputs"]:
        if not isinstance(source, dict) or not source.get("name") or not _hash(source.get("sha256"), 64):
            raise ValueError("each input needs a name and SHA256")
    media = meta.get("media")
    if not isinstance(media, list) or not media or media[0] != medium():
        raise ValueError("medium 0 must be explicit vacuum")
    for item in media[1:]:
        if not isinstance(item, dict) or not isinstance(item.get("material"), dict):
            raise ValueError("material description is required")
        if not item["material"].get("name") or item.get("loss_basis") not in ("restricted", "unrestricted"):
            raise ValueError("material identity and loss basis are required")
        description = item["material"]
        if set(description) != set(Material.__dataclass_fields__):
            raise ValueError("complete core Material description is required")
        for field in ("rho", "I_eV", "birks_mm_per_MeV"):
            value = description[field]
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"invalid material {field}")
        if description["rho"] == 0 or description["I_eV"] == 0:
            raise ValueError("material density and excitation energy must be positive")
        fractions = description["fractions"]
        if (not isinstance(fractions, dict) or not fractions or not set(fractions) <= set(ELEMENTS)
                or any(not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0 for v in fractions.values())
                or not np.isclose(sum(fractions.values()), 1., rtol=1e-6, atol=1e-8)):
            raise ValueError("invalid material mass fractions")
        density = description["sternheimer"]
        if (not isinstance(density, list) or len(density) not in (0, 6)
                or any(not isinstance(v, (int, float)) or not np.isfinite(v) for v in density)):
            raise ValueError("invalid Sternheimer parameters")
        cuts = description["cuts_e_MeV"]
        if (not isinstance(cuts, dict)
                or any(not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0 for v in cuts.values())):
            raise ValueError("invalid material region cuts")
        cut = item.get("cut_MeV")
        if not isinstance(cut, (int, float)) or not np.isfinite(cut) or cut <= 0:
            raise ValueError("effective cut must be positive")
    if meta.get("materials_cuts_sha256") != _digest(media):
        raise ValueError("material/cut identity mismatch")


class SegmentHistory:
    """Owned read-only columns, ordered by full muon identity then contiguous segment ordinal.

    Construction consumes already-produced samples and states. Unknown quantities are NaN with a status,
    never zero-filled. Validation certifies the representation, not the producer's physics or geometry.
    """

    def __init__(self, columns, metadata):
        self._metadata = _json_copy(metadata)
        _validate_manifest(self._metadata)
        if set(columns) != set(FIELDS):
            raise ValueError("segment columns must match the schema exactly")
        self._columns = {}
        for name in FIELDS:
            raw = np.asarray(columns[name])
            if name in INTEGER and (raw.dtype.kind not in "iu" or (raw > np.iinfo(np.int64).max).any()):
                raise ValueError(f"{name} must contain int64-compatible integers")
            dtype = np.int64 if name in INTEGER else str if name in TEXT else float
            self._columns[name] = np.array(raw, dtype=dtype, copy=True)
        self._validate()
        for name, array in self._columns.items():
            self._columns[name] = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)

    def __len__(self):
        return len(self._columns["run"])

    def __getitem__(self, name):
        view = self._columns[name].view()
        view.flags.writeable = False
        return view

    @property
    def metadata(self):
        return _json_copy(self._metadata)

    @classmethod
    def empty(cls, metadata):
        """Return all schema fields with zero rows, including (0, 3) position arrays."""
        columns = {name: np.empty((0, 3) if name in VECTOR else (0,),
                                 dtype=np.int64 if name in INTEGER else str if name in TEXT else float)
                   for name in FIELDS}
        return cls(columns, metadata)

    def _validate(self):
        c = self._columns
        if c["run"].ndim != 1:
            raise ValueError("columns must have one row axis")
        n = len(self)
        for name in FIELDS:
            if c[name].shape != ((n, 3) if name in VECTOR else (n,)):
                raise ValueError(f"wrong shape for {name}")
        for name in KEY:
            if (c[name] < 0).any():
                raise ValueError("keys must be nonnegative")
        for name in ("path_mm", "ke_in_MeV", "ke_out_MeV", "applied_loss_MeV") + VECTOR:
            if not np.isfinite(c[name]).all() or (name not in VECTOR and (c[name] < 0).any()):
                raise ValueError(f"{name} must be finite and nonnegative (positions may be signed)")
        for value, status in (("sampled_loss_MeV", "sample_status"), ("deposited_MeV", "deposit_status"),
                              ("escaping_MeV", "escape_status"), ("dt_ns", "dt_status")):
            known = c[status] == "known"
            if not np.isin(c[status], STATUSES).all():
                raise ValueError(f"invalid {status}")
            if not (np.isfinite(c[value][known]) & (c[value][known] >= 0)).all():
                raise ValueError(f"known {value} must be finite and nonnegative")
            if not np.isnan(c[value][~known]).all():
                raise ValueError(f"unknown/unmodeled {value} must be NaN")
        if not np.isin(c["kind"], ("physical", "displacement")).all():
            raise ValueError("invalid segment kind")
        if not np.isin(c["terminal"], ("none", "exhausted", "model_stop")).all():
            raise ValueError("invalid terminal state")
        physical = c["kind"] == "physical"
        if ((c["medium"][physical] < 0) | (c["medium"][physical] >= len(self._metadata["media"]))).any():
            raise ValueError("physical segment needs a valid medium")
        if (c["medium"][~physical] != -1).any():
            raise ValueError("displacement has no physical medium (use -1)")
        delta = c["ke_in_MeV"] - c["ke_out_MeV"]
        if not np.allclose(delta, c["applied_loss_MeV"], rtol=1e-10, atol=1e-10):
            raise ValueError("primary kinetic debit does not close")
        if (c["ke_out_MeV"] > c["ke_in_MeV"]).any():
            raise ValueError("passive segments cannot gain kinetic energy")
        exhausted = c["terminal"] == "exhausted"
        if (c["ke_out_MeV"][exhausted] != 0).any() or ((c["ke_out_MeV"] == 0) & (c["terminal"] == "none")).any():
            raise ValueError("zero kinetic energy must be an absorbing exhausted/model stop")
        if ((c["ke_in_MeV"] == 0) & (c["path_mm"] > 0)).any():
            raise ValueError("stopped lanes cannot traverse physical path")
        chord = np.linalg.norm(c["exit_mm"] - c["entry_mm"], axis=1)
        if (chord[physical] > c["path_mm"][physical] + 1e-8).any():
            raise ValueError("physical path cannot be shorter than its chord")
        inert = ~physical | (c["medium"] == 0) | (c["path_mm"] == 0)
        for value, status in (("sampled_loss_MeV", "sample_status"), ("deposited_MeV", "deposit_status"),
                              ("escaping_MeV", "escape_status")):
            if ((c[status][inert] != "known") | (c[value][inert] != 0)).any():
                raise ValueError("vacuum, zero length and displacement require known zero energy transfer")
        if (c["applied_loss_MeV"][inert] != 0).any():
            raise ValueError("vacuum, zero length and displacement cannot debit energy")
        instantaneous = ~physical | (c["path_mm"] == 0)
        if (c["path_mm"][~physical] != 0).any() or ((c["dt_status"][instantaneous] != "known") |
                                                       (c["dt_ns"][instantaneous] != 0)).any():
            raise ValueError("displacement and zero length have zero physical flight time")
        if ((c["path_mm"] > 0) & (c["dt_status"] == "known") & (c["dt_ns"] == 0)).any():
            raise ValueError("positive physical path needs positive flight time or explicit unknown status")
        subtotal = (np.where(c["deposit_status"] == "known", c["deposited_MeV"], 0.)
                    + np.where(c["escape_status"] == "known", c["escaping_MeV"], 0.))
        if (subtotal > c["applied_loss_MeV"] + 1e-10 * (1 + c["applied_loss_MeV"])).any():
            raise ValueError("known partition subtotal exceeds primary debit")
        known = (c["deposit_status"] == "known") & (c["escape_status"] == "known")
        if not np.allclose((c["deposited_MeV"] + c["escaping_MeV"])[known],
                           c["applied_loss_MeV"][known], rtol=1e-10, atol=1e-10):
            raise ValueError("declared complete energy partition does not close")
        if n == 0:
            return
        order = np.lexsort(tuple(c[name] for name in reversed(KEY)))
        if not np.array_equal(order, np.arange(n)):
            raise ValueError("rows must be ordered by full identity and segment")
        same = np.logical_and.reduce([c[name][1:] == c[name][:-1] for name in KEY[:-1]])
        first = np.r_[True, ~same]
        if (c["segment"][first] != 0).any() or (np.diff(c["segment"])[same] != 1).any():
            raise ValueError("segment ordinals must start at zero and be contiguous per muon")
        if (same & (c["terminal"][:-1] != "none")).any():
            raise ValueError("stop is absorbing: no later segment for that muon")
        if not np.allclose(c["ke_in_MeV"][1:][same], c["ke_out_MeV"][:-1][same], rtol=1e-10, atol=1e-10):
            raise ValueError("discontinuous kinetic history")
        if not np.allclose(c["entry_mm"][1:][same], c["exit_mm"][:-1][same], rtol=0, atol=1e-8):
            raise ValueError("discontinuous positions: record displacement jumps explicitly")

    def accounting(self):
        """Return primary debit and partition residual; incomplete partitions stay NaN."""
        c = self._columns
        complete = (c["deposit_status"] == "known") & (c["escape_status"] == "known")
        return {"primary_residual_MeV": c["ke_in_MeV"] - c["ke_out_MeV"] - c["applied_loss_MeV"],
                "partition_complete": complete,
                "partition_residual_MeV": np.where(complete, c["applied_loss_MeV"] - c["deposited_MeV"]
                                                    - c["escaping_MeV"], np.nan)}

    def save(self, path):
        """Persist one batch and manifest without pickle; refuse to overwrite any existing artifact."""
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, manifest_json=np.asarray(json.dumps(self._metadata, sort_keys=True)),
                                **self._columns)

    @classmethod
    def load(cls, path):
        """Load and revalidate an NPZ batch and embedded JSON manifest."""
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(FIELDS) | {"manifest_json"}:
                raise ValueError("unexpected persisted schema")
            return cls({name: saved[name] for name in FIELDS}, json.loads(str(saved["manifest_json"])))


def capture(columns, metadata, *, enabled=False):
    """Optional passive capture; disabled means no inspection, allocation or random draws."""
    return SegmentHistory(columns, metadata) if enabled else None
