"""Off-spill same-step raw-digi comparison; run with python -m trkfastsim.validation.run1a_digi.

Requires the complete lossless trkdigi/events census, including empty events.
No hit reconstruction or nominal-field substitution. Histogram edge bins include
under/overflow; uncertainties pair events, not individual digis or ADC samples.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from trkfastsim.digi import digicond, fastdigi
from trkfastsim.validation.digicmp import LSB_TDC, ev_bin_counts, paired_var
from trkfastsim.validation import run1a_digi_io as io

KEYS = ("run", "subrun", "event")
FIELDS = ("sid", "tdc0", "tdc1", "tot0", "tot1", "pmp", "nadc", "adc", "flags")


def key_ids(keys):
    """Full tuple identities, restricted to the lossless ROOT writer's uint32 domain."""
    k = io.array(keys, "event keys")
    if (k.ndim != 2 or k.shape[1] != 3 or k.dtype.kind not in "iu"
            or np.any(k < 0) or np.any(k > np.iinfo(np.uint32).max)):
        raise ValueError("event keys must be uint32-domain integer run/subrun/event triples")
    return fastdigi._full_keyid(k)


def read_reference(path):
    return io.read_table(path)


def restrict_steps(steps, keys):
    """Explicit census membership; never infer empty-event coverage from step rows."""
    kid = key_ids(keys)
    if len(np.unique(kid)) != len(kid):
        raise ValueError("duplicate reference event keys")
    sid = key_ids(steps[list(KEYS)].to_numpy())
    keep = np.isin(sid, kid)
    counts = {"source_steps": len(steps), "source_step_events": len(np.unique(sid)),
              "selected_steps": int(keep.sum()), "excluded_steps": int((~keep).sum()),
              "empty_input_events": len(keys) - len(np.unique(sid[keep]))}
    return steps.loc[keep].reset_index(drop=True), counts


def field_at_origin(path, origin):
    """BFieldPlotter CSV x,y,z,Bx,By,Bz[,|B|]; exact printed coordinates, never nearest."""
    points = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    origin = np.asarray(origin, float)
    if (origin.shape != (3,) or not np.isfinite(origin).all() or points.shape[1] not in (6, 7)
            or not np.isfinite(points).all()):
        raise ValueError("field CSV requires finite x,y,z,Bx,By,Bz samples and a finite origin")
    at = np.all(points[:, :3] == origin, axis=1)
    if np.count_nonzero(at) != 1:
        raise ValueError("field CSV must contain exactly one sample at the EXACT detector origin")
    return float(points[at, 5][0])


def comparison(stock, digi, nevents, window_ns=100000., tdc_lsb_ns=LSB_TDC):
    """Sparse event-clustered histograms; paired yield ratio uses delta-method SE."""
    io.validate_comparison(stock, digi, nevents, window_ns, tdc_lsb_ns)
    fast = {"ev": digi["ev"], "sid": digi["sid"], "pmp": digi["pmp"], "adc": digi["adc"].ravel(),
            "nadc": np.full(len(digi["sid"]), digi["adc"].shape[1])}
    fast.update({f"{name}{i}": digi[name][:, i] for name in ("tdc", "tot") for i in range(2)})
    s, f = [np.bincount(d["ev"], minlength=nevents).astype(float) for d in (stock, fast)]
    ratio = float(f.sum() / s.sum()) if s.sum() else None
    se = float(np.std(f - ratio * s, ddof=1) / np.sqrt(nevents) / s.mean()) if ratio is not None and nevents > 1 else None
    result = {"yield": {"events": nevents, "stock_total": int(s.sum()), "fast_total": int(f.sum()),
                        "stock_mean": float(s.mean()), "fast_mean": float(f.mean()), "ratio": ratio,
                        "ratio_se": se, "empty_stock_events": int((s == 0).sum()), "empty_fast_events": int((f == 0).sum())}}
    edges = {"plane": np.arange(37) - .5, "tdc0": np.linspace(0, window_ns, 101),
             "tdc1": np.linspace(0, window_ns, 101), "tot0": np.arange(17) - .5,
             "tot1": np.arange(17) - .5, "pmp": np.linspace(0, 1500, 31), "adc": np.linspace(0, 4096, 33)}
    for name, bins in edges.items():
        hs = []
        for d in (stock, fast):
            values = (d["sid"].astype(np.int64) >> 10) if name == "plane" else d[name]
            if name.startswith("tdc"):
                values = values * tdc_lsb_ns
            ev = np.repeat(d["ev"], d["nadc"]) if name == "adc" else d["ev"]
            hs.append(ev_bin_counts(values, ev, np.r_[-np.inf, bins, np.inf]))
        a, b = hs
        raw = paired_var(b, a, len(bins) + 1)
        variance = np.maximum(0, nevents / (nevents - 1) * (raw - (b[0] - a[0]) ** 2 / nevents)) if nevents > 1 else None
        difference = b[0] - a[0]
        valid = variance > 0 if variance is not None else np.zeros(len(difference), bool)
        pulls = np.divide(difference, np.sqrt(variance), out=np.zeros_like(difference, dtype=float), where=valid) if variance is not None else None
        result[name] = {"edges": bins.tolist(), "includes_flow_bins": True,
                        "edge_convention": "left closed, right open; flow bins first/last",
                        "units": "ns" if name.startswith("tdc") else "plane index" if name == "plane" else "ADC counts" if name in ("adc", "pmp") else "TOT ticks",
                        "stock_total": a[0].tolist(), "fast_total": b[0].tolist(),
                        "stock_sum_event_sq": a[1].tolist(), "fast_sum_event_sq": b[1].tolist(),
                        "paired_sum_difference_sq": raw.tolist(), "paired_total_variance": None if variance is None else variance.tolist(),
                        "chi2_diagonal": float(np.sum(pulls[valid] ** 2)) if pulls is not None else None,
                        "ndf_diagonal": int(valid.sum()), "zero_variance_different_bins": int(np.count_nonzero((difference != 0) & ~valid))}
    return result


def refresh_report(path, output):
    """Read saved products and write a NEW report; never rewrite producing evidence."""
    path, output = Path(path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    report = json.loads(path.read_text())
    reference_path = Path(report["inputs"]["reference"])
    if not reference_path.is_absolute():
        raise ValueError("legacy reference path must be absolute; relative provenance is ambiguous")
    paths = {"reference": reference_path, "fast": path.parent / "fast.root",
             "conditions": path.parent / "conditions.npz"}
    expected = dict(report.get("outputs_sha256", {}))
    if "reference" in report.get("inputs_sha256", {}):
        expected["reference"] = report["inputs_sha256"]["reference"]
    actual = io.verify_hashes(expected, paths)
    keys, stock = read_reference(paths["reference"])
    fastkeys, raw = io.read_table(paths["fast"], "trkdump/events", require_flags=False)
    np.testing.assert_array_equal(keys, fastkeys)
    with np.load(paths["conditions"], allow_pickle=False) as saved:
        scalars = json.loads(str(saved["__scalars__"]))
    digi = io.rectangular(raw, scalars["nADCSamples"])
    census = {"reference_events": len(keys), "reference_digis": len(stock["sid"]),
              "reference_adc_samples": len(stock["adc"]), "fast_digis": len(raw["sid"]),
              "fast_adc_samples": len(raw["adc"])}
    for name, value in census.items():
        if name in report and report[name] != value:
            raise ValueError(f"saved report census differs: {name}")
    report["paired"] = comparison(stock, digi, len(keys), scalars["eventWindowLength"], scalars["TDCLSB"])
    report["refresh"] = {"source_report_sha256": io.sha256(path), "consumed_sha256": actual,
                         "validation_source_sha256": io.hashes({"driver": __file__, "io": io.__file__}),
                         "verified_producing_hashes": sorted(expected),
                         "scope": "current saved bytes only; unbound historical provenance is not repaired"}
    io.verify_hashes(actual, paths)
    io.write_report(output, report)
    print(json.dumps({"yield": report["paired"]["yield"], "shapes": {
        k: {f: v[f] for f in ("chi2_diagonal", "ndf_diagonal", "zero_variance_different_bins")}
        for k, v in report["paired"].items() if k != "yield"}}, indent=2))
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--refresh-report":
        refresh = argparse.ArgumentParser(description="Recompare saved raw digis without digitising again")
        refresh.add_argument("--refresh-report", type=Path, required=True)
        refresh.add_argument("--output", type=Path, required=True)
        args = refresh.parse_args(argv)
        return refresh_report(args.refresh_report, args.output)
    if argv and argv[0] == "--stock-control":
        control = argparse.ArgumentParser(description="Compare two saved stock seeds and check disjoint event blocks")
        control.add_argument("--stock-control", type=Path, required=True)
        control.add_argument("--reference", type=Path, required=True)
        control.add_argument("--heldout", type=Path, required=True)
        control.add_argument("--output", type=Path, required=True)
        control.add_argument("--window-ns", type=float, default=100000.)
        control.add_argument("--tdc-lsb-ns", type=float, default=LSB_TDC)
        args = control.parse_args(argv)
        if args.output.exists():
            raise FileExistsError(args.output)
        inputs = {k: v.resolve() for k, v in vars(args).items() if isinstance(v, Path) and k != "output"}
        inputs_sha256 = io.hashes(inputs)
        keys, stock = read_reference(args.reference)
        other, raw = read_reference(args.stock_control)
        holdkeys, _ = read_reference(args.heldout)
        np.testing.assert_array_equal(keys, other)
        if np.intersect1d(key_ids(keys), key_ids(holdkeys)).size:
            raise ValueError("development and held-out event blocks overlap")
        samples = np.unique(raw["nadc"])
        if len(samples) > 1 or (len(samples) and samples[0] <= 0):
            raise ValueError("control needs a constant waveform length")
        digi = io.rectangular(raw, int(samples[0]) if len(samples) else 1)
        report = dict(schema_version=1, inputs={k: str(v) for k, v in inputs.items()}, inputs_sha256=inputs_sha256,
                      development_events=len(keys), heldout_events=len(holdkeys), disjoint=True,
                      window_ns=args.window_ns, tdc_lsb_ns=args.tdc_lsb_ns,
                      paired=comparison(stock, digi, len(keys), args.window_ns, args.tdc_lsb_ns))
        io.verify_hashes(inputs_sha256, inputs)
        io.write_report(args.output, report)
        print(json.dumps(report["paired"]["yield"], indent=2))
        return report
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("steps", "reference", "config", "preamp", "align", "gdml", "global-constants", "field-csv", "outdir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--delay-panel", type=Path)
    parser.add_argument("--delay-straw", type=Path)
    parser.add_argument("--detector-origin", type=float, nargs=3, required=True)
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)
    if args.chunk < 1 or args.repeats < 1 or args.seed < 0:
        parser.error("chunk/repeats must be positive and seed nonnegative")
    if (args.delay_panel is None) != (args.delay_straw is None):
        parser.error("--delay-panel and --delay-straw must be supplied together")
    if args.require_gpu and args.backend != "jax":
        parser.error("--require-gpu requires --backend jax")
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    inputs = {k: v.resolve() for k, v in vars(args).items() if isinstance(v, Path) and k != "outdir"}
    inputs_sha256 = io.hashes(inputs)
    begin = time.perf_counter()
    cfg = digicond.parse_fhicl_dump(args.config.read_text())
    if digicond.fcl_get(cfg, "physics.producers.EWMProducer.SpillType") != 0:
        raise ValueError("this driver requires off-spill EWMProducer.SpillType=0")
    bz = field_at_origin(args.field_csv, args.detector_origin)
    cond = digicond.build(args.config, args.preamp, args.gdml, args.align,
                         delay_csvs=(args.delay_panel, args.delay_straw) if args.delay_panel else None,
                         global_constants=args.global_constants, field_bz_T=bz, detector_origin=args.detector_origin)
    if cond.onspill:
        raise ValueError("this driver requires off-spill conditions")
    keys, stock = read_reference(args.reference)
    steps, counts = restrict_steps(fastdigi.load_steps(args.steps), keys)
    devices = ["NumPy host"]
    if args.backend == "jax":
        import jax
        devices = [str(d) for d in jax.devices()]
        selected_device = getattr(getattr(jax, "config", None), "jax_default_device", None) or jax.devices()[0]
        if args.require_gpu and (jax.default_backend() != "gpu" or selected_device.platform != "gpu"):
            raise RuntimeError("GPU requested but default backend is not GPU")
        devices = [str(selected_device)]
    args.outdir.mkdir(parents=True, exist_ok=False)
    cond.save(args.outdir / "conditions.npz")
    setup_s = time.perf_counter() - begin
    passes, first = [], None
    for index in range(args.repeats):
        rng = np.random.default_rng(args.seed) if args.backend == "numpy" else jax.random.key(args.seed)
        stats, begin = {"numpy_fallback_chunks": 0}, time.perf_counter()
        outkeys, digi, nsteps = fastdigi.digitize_steps(steps, cond, rng, keys=keys, chunk=args.chunk,
                                                     stats=stats, backend=args.backend)
        elapsed = time.perf_counter() - begin  # both backends return synchronized host arrays
        if not np.array_equal(keys, outkeys) or nsteps != len(steps):
            raise ValueError("digitiser changed event population or step count")
        if first is None:
            first = {k: v.copy() for k, v in digi.items()}
        elif first.keys() != digi.keys() or any(first[k].dtype != digi[k].dtype or not np.array_equal(first[k], digi[k], equal_nan=True) for k in first):
            raise ValueError("same-seed repeat changed output arrays")
        passes.append({"pass": index, "processing_s": elapsed, "events_per_s": len(keys) / elapsed, "stats": stats})
        print(json.dumps(passes[-1]), flush=True)
    io.validate_comparison(stock, first, len(keys), cond.eventWindowLength, cond.TDCLSB)
    begin = time.perf_counter()
    fastdigi.write_digis(args.outdir / "fast.root", keys, first, cond.nADCSamples)
    write_s = time.perf_counter() - begin
    io.verify_hashes(inputs_sha256, inputs)
    report = {"schema_version": 1, "inputs": {k: str(v) for k, v in inputs.items()},
              "inputs_sha256": inputs_sha256,
              "outputs_sha256": io.hashes({"fast": args.outdir / "fast.root", "conditions": args.outdir / "conditions.npz"}),
              "validation_source_sha256": io.hashes({"driver": __file__, "io": io.__file__}),
              "reference_events": len(keys), "reference_digis": len(stock["sid"]), "reference_adc_samples": len(stock["adc"]),
              **counts, "fast_digis": len(first["sid"]), "fast_adc_samples": int(first["adc"].size),
              "event_window_ns": cond.eventWindowLength, "backend": args.backend, "devices": devices, "seed": args.seed, "chunk": args.chunk,
              "detector_origin_mm": args.detector_origin, "field_bz_T": bz, "setup_s": setup_s, "root_write_s": write_s,
              "passes": passes, "repeat_identity": True if args.repeats > 1 else None, "paired": comparison(stock, first, len(keys), cond.eventWindowLength, cond.TDCLSB),
              "boundary": "off-spill steps to raw digis; pbtime=0, full conditions window; no hit reco or full StrawDigiMC",
              "truth_rows": "d_step indexes the census-filtered step table; no original-row remapping"}
    io.write_report(args.outdir / "report.json", report)
    return report


if __name__ == "__main__":
    main()
