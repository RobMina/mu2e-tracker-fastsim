"""Run 1A off-spill digi -> SH/PH closure; no truth, tracking, or tolerance verdicts."""
import argparse
import json
import os
import time
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from trkfastsim.reco import combo, hitreco

SH_COLS = tuple("time et0 et1 tot0 tot1 edep eend dtime ptime wdist uvar vvar timevar x y z flag".split())
PH_COLS = tuple("edep et0 et1 time wdist x y z uvar flag".split())
DIGI_COLS = tuple("sid tdc0 tdc1 tot0 tot1 pmp nadc adc".split())
INTEGERS = {"ev", "sid", "flag", "eend", "nsh", "ncombo", "ind"}
OFFSPILL = 0  # Offline/DataProducts/inc/EventWindowMarker.hh: offspill=0,onspill=1


def _integer(a, name):
    if a.dtype.kind not in "iu" or np.any(a < 0):
        raise ValueError(f"{name} must contain nonnegative integers")


def decode_reference(rows, npre):
    """Validate the complete census and jagged payload before flattening any alignment."""
    keys = np.column_stack([ak.to_numpy(rows[k]) for k in ("run", "subrun", "event")])
    _integer(keys, "event keys")
    if np.any(keys > np.iinfo(np.uint32).max) or len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("invalid or duplicate full event keys")
    pbt, spill = (ak.to_numpy(rows[k]) for k in ("pbt", "spill"))
    _integer(spill, "spill")
    if np.any(spill != OFFSPILL) or not np.isfinite(pbt).all() or np.any(pbt != 0):
        raise ValueError("Run1A offspill requires spill=0 and finite pbt=0")
    if isinstance(npre, (bool, np.bool_)) or not isinstance(npre, (int, np.integer)) or npre <= 0:
        raise ValueError("nADCPreSamples must be a positive integer")

    def counts(k):
        return ak.to_numpy(ak.num(rows[k], axis=1))

    def flat(k, expected=None, integer=False):
        if expected is not None and not np.array_equal(counts(k), expected):
            raise ValueError(f"{k} jagged lengths disagree")
        a = ak.to_numpy(ak.flatten(rows[k], axis=1))
        if not np.isfinite(a).all():
            raise ValueError(f"{k} contains nonfinite values")
        if integer:
            _integer(a, k)
        return a

    nd = counts("d_sid")
    d = {k: flat("d_" + k, None if k == "adc" else nd, True) for k in DIGI_COLS}
    if np.any(d["nadc"] <= npre):
        raise ValueError("waveform lengths must be positive and greater than npre")
    if not np.array_equal(ak.to_numpy(ak.sum(rows["d_nadc"], axis=1)), counts("d_adc")):
        raise ValueError("d_nadc does not account for every ADC sample per event")
    d.update(ev=np.repeat(np.arange(len(keys)), nd), pbt=pbt, evkey=keys)
    refs = []
    for pre, cols in (("s_", SH_COLS), ("p_", PH_COLS)):
        nh = counts(pre + "sid")
        names = set(("sid",) + cols + (("ncombo", "ind") if pre == "p_" else ()))
        names.update(k[2:] for k in rows.fields if k.startswith(pre))
        tab = {k: flat(pre + k, None if k == "ind" else nh, k in INTEGERS) for k in sorted(names)}
        tab["ev"] = np.repeat(np.arange(len(keys)), nh)
        if "ind" in tab:
            if np.any(tab["ncombo"] <= 0) or not np.array_equal(
                    ak.to_numpy(ak.sum(rows[pre + "ncombo"], axis=1)), counts(pre + "ind")):
                raise ValueError(f"{pre}ncombo does not account for constituents")
            limit = np.repeat(nd if pre == "s_" else counts("s_sid"), counts(pre + "ind"))
            if np.any(tab["ind"] >= limit):
                raise ValueError(f"{pre}ind constituent outside event collection")
        refs.append(tab)
    return d, refs


def column_stats(reference, candidate, integer=False):
    """Exact comparisons; residual summaries use finite pairs only, never a pass tolerance."""
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape:
        raise ValueError("column shapes differ")
    if integer:
        _integer(a, "reference column")
        _integer(b, "candidate column")
    finite = np.isfinite(a) & np.isfinite(b)
    if integer:
        ua, ub = a.astype(np.uint64), b.astype(np.uint64)
        equal = ua == ub
        delta = (np.maximum(ua, ub) - np.minimum(ua, ub)).astype(np.longdouble)
    else:
        equal = a == b
        delta = b[finite].astype(np.longdouble) - a[finite].astype(np.longdouble)
    exact = int(np.count_nonzero(finite & equal))
    maximum = np.max(np.abs(delta)) if delta.size else None
    rms = maximum * np.sqrt(np.mean((delta / maximum) ** 2)) if maximum else maximum
    safe = lambda x: float(x) if x is not None and np.isfinite(x) and x <= np.finfo(float).max else None
    return {"count": int(a.size), "integer": integer, "exact_count": exact,
            "exact_fraction": exact / a.size if a.size else None,
            "all_exact": bool(exact == a.size), "finite": bool(finite.all()),
            "reference_nonfinite": int(np.count_nonzero(~np.isfinite(a))),
            "candidate_nonfinite": int(np.count_nonzero(~np.isfinite(b))),
            "finite_pairs": int(finite.sum()), "maxabs": safe(maximum), "rms": safe(rms)}


def compare_hits(reference, candidate, panel=False):
    """Collection-order identity gates residuals; never compare misaligned SH or PH lists."""
    ids = ("ev", "sid", "ncombo", "ind") if panel else ("ev", "sid")
    identity = {k: bool(np.array_equal(reference[k], candidate[k])) for k in ids}
    aligned = all(identity.values())
    cols = PH_COLS if panel else SH_COLS
    return {"reference_count": len(reference["ev"]), "candidate_count": len(candidate["ev"]),
            "identity": identity, "aligned": aligned,
            "reference_finite": all(bool(np.isfinite(reference[k]).all()) for k in cols),
            "candidate_finite": all(bool(np.isfinite(candidate[k]).all()) for k in cols),
            "columns": {k: column_stats(reference[k], candidate[k], k in INTEGERS) for k in cols} if aligned else {},
            "numerical_skip_reason": None if aligned else "collection identity mismatch"}


def _jax_backend(require_gpu):
    import jax
    if require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("--require-gpu requires the active JAX backend to be GPU")
    from trkfastsim.reco import combo_jax, hitreco_jax
    return hitreco_jax, combo_jax, {"platform": jax.default_backend(), "devices": [str(d) for d in jax.devices()]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True, help="new directory (never overwritten)")
    parser.add_argument("--jax", action="store_true", help="also run the JAX SH+PH chain")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args(argv)
    if args.require_gpu and not args.jax:
        parser.error("--require-gpu requires --jax")
    timings = {}

    def timed(name, fn):
        start = time.perf_counter()
        result = fn()
        timings[name] = time.perf_counter() - start
        return result

    backends = [("numpy", hitreco, combo, {"platform": "cpu"})]
    if args.jax:
        sj, pj, info = timed("jax_import_setup_s", lambda: _jax_backend(args.require_gpu))
        backends.append(("jax", sj, pj, info))
    args.outdir.mkdir(parents=True, exist_ok=False)

    def read():
        with uproot.open(args.reference) as root:
            return root["trkdump/events"].arrays(library="ak")

    rows = timed("root_read_s", read)
    # The reused extractor writes to a sink; final condition-file writing is timed separately.
    rawcond = timed("condition_extraction_s", lambda: hitreco.extract_conditions(args.reference, os.devnull))
    cpath = args.outdir / "conditions.json"
    timed("conditions_write_s", lambda: cpath.write_text(json.dumps(rawcond, indent=1, allow_nan=False) + "\n"))
    cond = timed("condition_load_s", lambda: hitreco.load_conditions(cpath))
    d, refs = timed("input_validation_s", lambda: decode_reference(rows, cond["scalar"]["nADCPreSamples"]))
    us = hitreco.unique_straw(d["sid"].astype(np.int64))
    if (np.any((d["sid"] & 127) >= 96) or np.any(((d["sid"] >> 7) & 7) >= 6)
            or np.any(us >= len(cond["straw"]["hlen"]))):
        raise ValueError("d_sid outside extracted straw conditions")
    ne = len(d["evkey"])
    nd = np.bincount(d["ev"], minlength=ne)
    report = {"reference": str(args.reference.resolve()), "settings": "mc", "events": ne,
              "event_keys": d["evkey"].tolist(), "digis_per_event": nd.tolist(), "digis": len(d["ev"]),
              "zero_digi_events": int((nd == 0).sum()), "offspill": True, "pbt_finite_zero": True,
              "stock_negative_energy": {p: int((r["edep"] < 0).sum()) for p, r in zip(("sh", "ph"), refs)},
              "timings_s": timings, "backends": {},
              "timing_scope": "Wall seconds; kernel wrappers include packing, transfers and host materialisation. "
                              "Condition extraction includes its ROOT read and sink write. No tolerance verdict."}
    outputs = {}
    for name, shr, phr, info in backends:
        kwargs = {}
        if name == "jax":
            def prepare():
                # pack_digis cannot reduce an empty ADC layout: supply one masked padding row.
                source = d if len(d["ev"]) else dict(d, ev=np.array([0]), sid=np.array([0]),
                    tdc0=np.array([0]), tdc1=np.array([0]), tot0=np.array([0]), tot1=np.array([0]),
                    pbt=np.array([0.]), nadc=np.array([cond["scalar"]["nADCPreSamples"] + 1]),
                    adc=np.zeros(cond["scalar"]["nADCPreSamples"] + 1, dtype=int))
                packed = shr.pack_digis(source, cond["scalar"]["nADCPreSamples"])
                packed["n"], packed["valid"][len(d["ev"]):] = len(d["ev"]), False
                return {"dcond": shr.device_conditions(cond), "packed": packed}
            kwargs = timed("jax_condition_pack_s", prepare)
        first_sh = timed(name + "_sh_first_s", lambda: shr.straw_hits(d, cond, **kwargs))
        first_ph = timed(name + "_ph_first_s", lambda: phr.panel_hits(first_sh))
        sh = timed(name + "_sh_repeat_s", lambda: shr.straw_hits(d, cond, **kwargs))
        ph = timed(name + "_ph_repeat_s", lambda: phr.panel_hits(sh))
        tabs = (sh, ph)
        result = {"backend": info, "negative_energy": {p: int((t["edep"] < 0).sum()) for p, t in zip(("sh", "ph"), tabs)},
                  "stock_comparison": {p: compare_hits(r, t, p == "ph") for p, r, t in zip(("sh", "ph"), refs, tabs)},
                  "repeat": {p: compare_hits(r, t, p == "ph") for p, r, t in zip(("sh", "ph"), (first_sh, first_ph), tabs)},
                  "counts_per_event": {p: np.bincount(t["ev"], minlength=ne).tolist() for p, t in zip(("sh", "ph"), tabs)}}
        if name == "jax":
            result["numpy_comparison"] = {p: compare_hits(r, t, p == "ph") for p, r, t in zip(("sh", "ph"), outputs["numpy"], tabs)}
        report["backends"][name], outputs[name] = result, tabs
        payload = {p + "_" + k: v for p, t in zip(("s", "p"), tabs) for k, v in t.items()}
        timed(name + "_write_s", lambda: np.savez_compressed(args.outdir / (name + ".npz"), evkey=d["evkey"], **payload))
    path = args.outdir / "report.json"
    timed("report_write_s", lambda: path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n"))
    # Final metadata update is excluded from report_write_s (the measured full initial report write).
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(path)
    return report


if __name__ == "__main__":
    main()
