"""F1 closure report: yield exchange (gains/losses) by mechanism and fixed-stock-state fidelity, per vr2f1 arm.

Inputs: arm directories written by `reco vr2f1 --save-yield-ids` (ids.npz, vr2f1.json), the stock
KinematicLineFitTrace ROOT file of the same events (final KinKal status history per seed), and optionally a saved
paired.npz of earlier F1 arms (keys run/subrun/event/seed and *_good masks) to cross-tabulate the same identities.
All comparisons are stock-conditioned: F1 uses the stock seed, hit candidates and crossing candidates.
"""
import argparse
import json
from pathlib import Path

import numpy as np

STATUS = {-1: "unfit", 0: "converged", 1: "unconverged", 2: "lowNDOF", 3: "gapdiverged", 4: "paramsdiverged",
          5: "chisqdiverged", 6: "outsidemap", 7: "failed", 8: "unphysical"}
NSEED = 4                       # CHSEEDFIT meta-iterations; the rest are CHDRIFTEXT
KL = ("d0", "phi0", "z0", "theta", "t0", "mom")


def stock_history(trace_path):
    """Final status, its meta-iteration (fit stage, construction row removed) and the rejection flag per seed 0."""
    import awkward as ak
    import uproot
    with uproot.open(trace_path) as f:
        parent = f["kktrace"]
        cycles = [k for k in parent.keys(cycle=True, recursive=False) if k.split(";")[0] == "trace"]
        key = max(cycles, key=lambda k: int(k.rsplit(";", 1)[1]))
        t = parent[key].arrays(["run", "subrun", "event", "iseed", "goodseed", "hstatus", "hmiter", "rejected",
                                "full_schedule", "stage", "maxit"])
    runs = np.unique(ak.to_numpy(t.run))
    if len(runs) != 1 or not bool(ak.all(t.full_schedule & (t.stage == -1) & (t.maxit == -1))):
        raise ValueError("closure report requires one run and full, uncapped stock histories")
    t = t[ak.to_numpy(t.iseed) == 0]
    fitrows = t.hstatus >= 0
    st = ak.to_numpy(ak.fill_none(ak.firsts(t.hstatus[fitrows][:, ::-1]), -1)).astype(int)
    mi = ak.to_numpy(ak.fill_none(ak.firsts((t.hmiter[fitrows] // 1000)[:, ::-1]), -1)).astype(int)
    off = ak.to_numpy(ak.fill_none(ak.min(t.hmiter[fitrows] // 1000, axis=1), 0)).astype(int)
    return {"key": np.column_stack([ak.to_numpy(t.subrun), ak.to_numpy(t.event)]).astype(np.int64),
            "status": st, "stage": mi - off, "goodseed": ak.to_numpy(t.goodseed).astype(bool),
            "rejected": ak.to_numpy(t.rejected).astype(bool), "entries": len(t), "run": int(runs[0]),
            "tree": "kktrace/" + key}


def match(a, b):
    """Vectorized identity join; both single-seed populations must have unique keys."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("incompatible identity columns")
    dtype = np.dtype([(f"k{i}", np.int64) for i in range(a.shape[1])])
    av = np.ascontiguousarray(a).view(dtype).ravel()
    bv = np.ascontiguousarray(b).view(dtype).ravel()
    if len(np.unique(av)) != len(av) or len(np.unique(bv)) != len(bv):
        raise ValueError("duplicate identities")
    if not len(bv):
        return np.full(len(av), -1, dtype=np.int64)
    order = np.argsort(bv)
    pos = np.searchsorted(bv[order], av)
    safe = np.minimum(pos, len(bv) - 1)
    return np.where((pos < len(bv)) & (bv[order[safe]] == av), order[safe], -1)


def first_bad_stage(stat):
    """First stage whose status is unusable (>= lowNDOF), -1 if none."""
    bad = stat >= 2
    return np.where(bad.any(1), bad.argmax(1), -1)


def exchange(good, stockok):
    g, l = int((good & ~stockok).sum()), int((~good & stockok).sum())
    ns = int(stockok.sum())
    return {"n": int(len(good)), "stock_good": ns, "f1_good": int(good.sum()), "both": int((good & stockok).sum()),
            "gains": g, "losses": l, "net": g - l, "net_se_mcnemar": float(np.sqrt(g + l)),
            "net_rel": (g - l) / ns if ns else None, "net_rel_se": float(np.sqrt(g + l)) / ns if ns else None,
            "exchanged_rel": (g + l) / ns if ns else None,
            "acceptance_agreement": float(np.mean(good == stockok)) if len(good) else None,
            "acceptance_agreement_target": 0.995}


def classify(ids, stock, rows):
    """Mechanism label per track: F1 side for losses, stock side for gains."""
    stat = ids["stat"]
    fb = first_bad_stage(stat)
    fstat = np.where(fb >= 0, stat[np.arange(len(stat)), np.maximum(fb, 0)], stat[:, -1])
    sst = np.where(rows >= 0, stock["status"][np.maximum(rows, 0)], -2)
    sstage = np.where(rows >= 0, stock["stage"][np.maximum(rows, 0)], -2)
    def labels(status, stage):
        out = np.full(status.shape, "unknown", dtype="U48")
        out[status < 2] = "envelope"
        for code, name in STATUS.items():
            if code >= 2:
                out[status == code] = name
        out = np.char.add(out, np.where(status < 2, "", np.where(stage < NSEED, "@seed", "@ext")))
        return np.where(status == -2, "no_trace", out)
    lab_f1 = labels(fstat, fb)
    lab_st = labels(sst, sstage)
    return lab_f1, lab_st, fb, fstat


def counts(labels, mask):
    u, c = np.unique(labels[mask], return_counts=True)
    return {str(k): int(v) for k, v in sorted(zip(u, c), key=lambda kv: -kv[1])}


def fidelity(ids):
    """Fixed-state T-VR2: (F1 - stock) / stock error on the t0 piece, FitOK tracks both keep."""
    import jax
    from trkfastsim.reco import kalman_jax as K
    kl = np.asarray(jax.jit(jax.vmap(K.kl_params))(ids["xt0"]))
    Jk = np.asarray(jax.jit(jax.vmap(jax.jacfwd(K.kl_params)))(ids["xt0"]))
    Ck = np.einsum("tij,tjk,tlk->til", Jk, ids["Ct0"], Jk)
    ok = ids["good"] & ids["stockok"]
    out = {"n_tracks": int(ok.sum()), "target_frac_within_0p1": 0.99,
           "population": "both accepted; nonfinite pulls fail rather than leave the denominator"}
    pulls = {}
    for i, k in enumerate(KL):
        dv = kl[:, i] - ids["stock_params"][:, i]
        if k == "phi0":
            dv = (dv + np.pi) % (2 * np.pi) - np.pi
        err = ids["stock_errors"][:, i]
        p = dv / np.where(err > 0, err, np.nan)
        m = ok & np.isfinite(p)
        pulls[k] = p
        n, w = int(ok.sum()), int((np.abs(p[m]) < 0.1).sum())
        f = w / n if n else None
        ratio = np.sqrt(np.maximum(Ck[:, i, i], 0)) / np.where(err > 0, err, np.nan)
        errok = ok & np.isfinite(ratio) & (Ck[:, i, i] >= 0)
        out[k] = {"n": n, "n_finite": int(m.sum()), "within_0p1": w, "frac_within_0p1": f,
                  "binomial_se": float(np.sqrt(f * (1 - f) / n)) if n else None,
                  "q16_50_84": np.quantile(p[m], [.16, .5, .84]).tolist() if m.any() else None,
                  "difference_q16_50_84": np.quantile(dv[m], [.16, .5, .84]).tolist() if m.any() else None,
                  "err_ratio_q50": float(np.median(ratio[errok])) if errok.any() else None,
                  "err_frac_within_2pct": float((errok & (np.abs(ratio - 1) < 0.02)).sum() / n) if n else None}
    geo = np.stack([np.abs(pulls[k]) for k in KL[:4]], 1)
    out["all_geometry_within_0p1"] = float(np.mean((geo < 0.1).all(1)[ok])) if ok.any() else None
    out["active_hits_identical"] = float(np.mean(((ids["fact"] == ids["stock_active"]) | ~ids["valid"]).all(1)[ok])) if ok.any() else None
    out["outlier_attribution"] = attribute_outliers(ids, (geo < 0.1).all(1), ok)
    return out, pulls, ok


def attribute_outliers(ids, inside, ok):
    """Geometry-within-0.1-sigma fraction split by the crossing activity F1 used in its last stage vs the stock flag,
    in bounds (between the first and last active hit on the t0 piece line) and anywhere."""
    if "xact_fit" not in ids:
        return None
    xv = ids["xvalid"].astype(bool)
    mis = (ids["xact_fit"].astype(bool) != ids["xact_stock"].astype(bool)) & xv
    mis_in = mis & ids["xinb"].astype(bool)
    hashit = ids["xhit"] >= 0
    out = {"membership_scope": "final-t0-line / stock-active-hit projection proxy, NOT actual iteration membership",
           "causal_attribution": False}
    for name, m in (("no_mismatch", ~mis.any(1)), ("mismatch_outside_proxy_only", mis.any(1) & ~mis_in.any(1)),
                    ("mismatch_inside_proxy", mis_in.any(1))):
        sel = ok & m
        out[name] = {"n": int(sel.sum()), "within_0p1": int((sel & inside).sum()),
                     "frac": float((sel & inside).sum() / sel.sum()) if sel.any() else None}
    bad = ok & ~inside
    out["outliers"] = int(bad.sum())
    out["outliers_with_inside_proxy_mismatch"] = int((bad & mis_in.any(1)).sum())
    mi = mis_in & ok[:, None]
    out["inside_proxy_mismatch_crossings"] = {"n": int(mi.sum()), "f1_only": int((mi & ids["xact_fit"]).sum()),
                                           "stock_only": int((mi & ids["xact_stock"]).sum()),
                                           "with_hit": int((mi & hashit).sum()),
                                           "abs_doca_q16_50_84": np.quantile(np.abs(ids["xdoca"][mi]), [.16, .5, .84]).tolist()
                                           if mi.any() else None}
    return out


def divergence_summary(ids, saved):
    """Front momentum term at the iteration with the largest endpoint chi2; not a joint-chi2 decomposition."""
    fb = first_bad_stage(ids["stat"])
    row, stage = np.arange(len(fb)), np.maximum(fb, 0)
    lost = ids["stockok"] & ~ids["good"] & (fb >= 0) & (ids["stat"][row, stage] == 4)
    out = {"n_paramsdiverged_losses": int(lost.sum()),
           "scope": "front marginal momentum term; maximum chi2 may be at the back, correlations not decomposed"}
    for key in ("dpmax", "mterm", "dmomf", "smref"):
        x = saved[key][row, stage]
        m = lost & np.isfinite(x)
        out[key] = {"n_finite": int(m.sum()),
                    "q16_50_84": np.quantile(x[m], [.16, .5, .84]).tolist() if m.any() else None}
    threshold = np.where(fb < NSEED, 1e8, 1e6)
    out["front_marginal_momentum_over_threshold"] = int((lost & (saved["mterm"][row, stage] > threshold)).sum())
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", action="append", required=True, help="NAME=DIR of a vr2f1 --save-yield-ids output")
    p.add_argument("--trace", required=True, help="stock KinematicLineFitTrace ROOT file of the same events")
    p.add_argument("--paired", default=None, help="saved paired.npz of earlier arms (keys, *_good)")
    p.add_argument("--expected-events", type=int, default=3000)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    stock = stock_history(a.trace)
    paired = dict(np.load(a.paired)) if a.paired else None
    rep = {"scope": "stock-conditioned F1: stock seed, hit and crossing candidates (audit T1); not an unconditional "
                    "reconstruction efficiency", "stock_trace": {"path": a.trace, "tree": stock["tree"],
                                                                "entries": stock["entries"]}, "arms": {}}
    tracks, checked_inputs, expected = {}, {}, {}
    for item in a.arm:
        name, d = item.split("=", 1)
        ids = dict(np.load(Path(d) / "ids.npz"))
        key = np.column_stack([ids["subrun"], ids["event"]]).astype(np.int64)
        rows = match(key, stock["key"])
        meta = json.loads((Path(d) / "vr2f1.json").read_text())
        if meta["input"] not in checked_inputs:
            import uproot
            with uproot.open(meta["input"]) as src:
                parent = src["trkdump"]
                keys = [k for k in parent.keys(cycle=True, recursive=False) if k.split(";")[0] == "events"]
                tree = max(keys, key=lambda k: int(k.rsplit(";", 1)[1]))
                entries = parent[tree].num_entries
                runs = np.unique(parent[tree]["run"].array(library="np"))
                if entries != a.expected_events or not np.array_equal(runs, [stock["run"]]):
                    raise ValueError("input event count/run differs from the frozen comparison")
                import awkward as ak
                arr = parent[tree].arrays(["subrun", "event", "sd_a0", "kl_d0", "kl_status"])
                eligible = ak.to_numpy((ak.num(arr.sd_a0) == 1) & (ak.num(arr.kl_d0) == 1))
                fitok = (ak.to_numpy(ak.fill_none(ak.firsts(arr.kl_status), 0)).astype(np.int64) & (1 << 15)) != 0
                ekeys = np.column_stack([ak.to_numpy(arr.subrun), ak.to_numpy(arr.event)]).astype(np.int64)
                expected[meta["input"]] = (ekeys, eligible, fitok)
                checked_inputs[meta["input"]] = {"tree": "trkdump/" + tree, "entries": entries, "run": int(runs[0]),
                                                "eligible_seeds": int(eligible.sum()), "eligible_stock_good": int((eligible & fitok).sum())}
        own = meta.get("own_states", name.startswith("own_"))
        yield_population = own or meta.get("all_fits", False)
        ekeys, eligible, fitok = expected[meta["input"]]
        selected = eligible if yield_population else eligible & fitok
        erows = match(key, ekeys[selected])
        if len(key) != int(selected.sum()) or np.any(erows < 0) or not np.array_equal(ids["stockok"], fitok[selected][erows]):
            raise ValueError(f"{name}: incomplete eligible population or changed stock acceptance")
        r = {"dir": d, "n_tracks": int(len(key)), "missing_trace": int((rows < 0).sum()),
             "configuration": {k: meta.get(k) for k in ("input", "own_states", "all_fits", "time_model",
                 "time_legacy", "xing_activity", "env_f1", "seed_time_provenance", "stock_exact",
                 "reference_model", "reference_unsupported_tracks", "crossing_time", "native_seed",
                 "construction_unsupported_tracks", "construction_unsupported_crossings", "constructed_crossings", "straw_geometry")},
             "numerical_validity": meta.get("numerical_validity")}
        if len(key) != meta["n_tracks"] or np.any(rows < 0):
            raise ValueError(f"{name}: incomplete metadata or missing stock identities")
        lab_f1, lab_st, fb, fstat = classify(ids, stock, rows)
        if yield_population:
            good, sok = ids["good"].astype(bool), ids["stockok"].astype(bool)
            r["yield"] = exchange(good, sok)
            r["loss_mechanism_f1"] = counts(lab_f1, sok & ~good)
            r["gain_mechanism_stock"] = counts(lab_st, good & ~sok)
            r["gain_stock_rejection_snapshot"] = int((good & ~sok & (rows >= 0) &
                                                      stock["rejected"][np.maximum(rows, 0)]).sum())
            same = ((ids["fact"] == ids["stock_active"]) | ~ids["valid"]).all(1)
            lr = np.where(ids["fact"], ids["flr"], 0) == np.where(ids["stock_active"], ids["stock_lr"], 0)
            same_lr = same & (lr | ~ids["valid"]).all(1)
            r["final_hit_states_identical_to_stock"] = {
                pop: {"n": int(m.sum()), "active_identical": int((m & same).sum()), "state_identical": int((m & same_lr).sum())}
                for pop, m in (("both_good", good & sok), ("gains", good & ~sok), ("losses", ~good & sok),
                               ("neither", ~good & ~sok))}
            r["stock_bad_mechanism"] = counts(lab_st, ~sok)
            r["f1_bad_mechanism"] = counts(lab_f1, ~good)
            r["own_state_fidelity_diagnostic"] = fidelity(ids)[0]
            if (Path(d) / "yield.npz").exists():
                with np.load(Path(d) / "yield.npz") as saved:
                    r["divergence_diagnostic"] = divergence_summary(ids, saved)
            if paired is not None:
                pk = paired["keys"][:, 1:3].astype(np.int64)
                pr = match(key, pk)
                r["paired_missing"] = int((pr < 0).sum())
                if np.any(pr < 0):
                    raise ValueError(f"{name}: missing paired identities")
                for arm in ("legacy", "candidate", "prior"):
                    og = paired[arm + "_good"][np.maximum(pr, 0)] & (pr >= 0)
                    r["vs_" + arm] = {"agree": int((og == good).sum()), "this_only": int((good & ~og).sum()),
                                      "that_only": int((~good & og).sum()),
                                      "that_losses_recovered": int((sok & ~og & good).sum()),
                                      "that_gains_kept": int((~sok & og & good).sum()),
                                      "that_losses": int((sok & ~og).sum()), "that_gains": int((~sok & og).sum())}
            tracks[name] = dict(key=key, good=good, stockok=sok, f1_label=lab_f1, stock_label=lab_st)
        else:
            fid, pulls, ok = fidelity(ids)
            r["fidelity"] = fid
            r["f1_failures_on_stock_fitok"] = counts(lab_f1, ~ids["good"].astype(bool))
        rep["arms"][name] = r
    rep["inputs"] = checked_inputs
    names = list(tracks)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            gx, gy = tracks[x]["good"], tracks[y]["good"]
            if len(gx) == len(gy) and (tracks[x]["key"] == tracks[y]["key"]).all():
                rep.setdefault("cross", {})[f"{x}->{y}"] = {"gains": int((~gx & gy).sum()), "losses": int((gx & ~gy).sum())}
    out = Path(a.out)
    out.write_text(json.dumps(rep, indent=1) + "\n")
    for name, t in tracks.items():
        ex = t["good"] != t["stockok"]
        table = np.column_stack([t["key"][ex], np.where(t["good"][ex], "gain", "loss"),
                                 t["f1_label"][ex], t["stock_label"][ex]])
        np.savetxt(out.with_name(out.stem + f"_{name}_exchange.csv"), table, fmt="%s", delimiter=",",
                   header="subrun,event,kind,f1_label,stock_label", comments="")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
