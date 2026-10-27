"""Audit actual world columns and plane transport before tracker acceptance.

python3 scripts/world_device_check.py --gen GEN.root --gdml WORLD.gdml --out NEW.json
All comparisons use primary muons as the statistical unit; no step-level independence assumption.
"""
import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from trkfastsim import geometry, steps as st, steps_jax as sj, tables, world_jax as wj


def summarize(a, b):
    a, b = np.asarray(a), np.asarray(b)
    delta = b - a
    return {"numpy_mean": float(a.mean()), "jax_mean": float(b.mean()), "paired_diff": float(delta.mean()),
            "paired_se": float(delta.std(ddof=1) / np.sqrt(len(a))),
            "numpy_quantiles": np.quantile(a, [.01, .1, .5, .9, .99]).tolist(),
            "jax_quantiles": np.quantile(b, [.01, .1, .5, .9, .99]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gen", "gdml", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-near", type=int, default=0)
    parser.add_argument("--columns-only", action="store_true")
    parser.add_argument("--check-untiled", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    gen = tables.read(args.gen, "gen")
    geom = geometry.from_gdml(args.gdml)
    world = st.load_world(args.gdml, geom)
    dev = wj.DeviceWorld(world, world.edges)
    mu = st._muons(gen, True)
    o = mu[["x", "y", "z"]].to_numpy()
    p = np.sqrt(mu.px ** 2 + mu.py ** 2 + mu.pz ** 2).to_numpy()
    d = mu[["px", "py", "pz"]].to_numpy() / p[:, None]
    near = np.flatnonzero(st.near_tracker(o, d, geom, st.plane_y(geom)))
    if args.max_near:
        near = near[:args.max_near]
    o, d, p = o[near], d[near], p[near]
    report = {"near_muons": len(p), "geometry": []}
    for k, (top, bot) in enumerate(zip(world.edges[:-1], world.edges[1:])):
        maxdiff, maxhit = 0.0, 0
        for s in range(0, len(p), 1024):
            oo, dd = jnp.asarray(o[s:s + 1024]), jnp.asarray(d[s:s + 1024])
            valid = jnp.ones(len(oo), bool)
            origin, length, _ = wj._band_segment(oo, dd, top, bot, world.edges[-1], valid)
            got = np.asarray(dev.column(k, origin, dd, length, valid))
            ps = world.band(k)
            ref = ps.column(np.asarray(origin), np.asarray(dd), np.asarray(length))
            if len(ps):
                h = int(wj._max_hits(dev.bands[k][0][0], dev.bands[k][0][1], origin, dd, length, valid))
                maxhit = max(maxhit, h)
                if args.check_untiled:
                    old, overflow = sj._calo_set(*dev.bands[k][0], 1, origin, dd, length, valid,
                                                 len(ps.mats), False, sj._bucket(h, 1))
                    assert not int(overflow)
                    np.testing.assert_array_equal(got, np.asarray(old / sj.QSCALE[0]))
            for m, v in ref.items():
                maxdiff = max(maxdiff, float(np.max(np.abs(got[:, ps.mats.index(m)] - v), initial=0)))
        report["geometry"].append({"band": k, "pieces": len(world.band(k)), "max_candidates": maxhit,
                                   "planes": world.band(k).P.shape[1], "materials": world.band(k).mats,
                                   "max_abs_column_diff_mm": maxdiff})
    if args.columns_only:
        report["untiled_exact_checked"] = args.check_untiled
        with out.open("x") as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report, indent=2))
        return
    rng = np.random.default_rng([1, 0])
    key = sj.key_from(np.random.default_rng([1, 0]))
    from mu2efastsim import eloss, materials, transport as tr
    t0 = time.perf_counter()
    ref = tr.world_loss(rng, o, d, p, world, world.edges, return_band_p=True, return_tail=True)
    pw, xx0, ok, bx, bh, bp, chi, tn = ref
    on, dn, ln = tr.msc_to_plane(rng, o, d, np.maximum(np.sqrt(p * np.maximum(pw, 1)), 1), world.edges[-1],
                                 eloss.radiation_length_mm(materials.AIR), xx0, bx, bh,
                                 band_p=bp, band_chic2=chi, band_thn=tn)
    report["numpy_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    got = dev.transport(key, o, d, p, indices=near)
    report["jax_compile_s"] = time.perf_counter() - t0
    both = ok & got[4]
    report["survived"] = [int(ok.sum()), int(got[4].sum())]
    report["survived_paired"] = summarize(ok.astype(float), got[4].astype(float))
    report["p_all"] = summarize(pw, got[2])
    report["p_survivors"] = summarize(pw[both], got[2][both])
    for k in range(3):
        report[f"position_{k}"] = summarize(on[both, k], got[0][both, k])
        report[f"direction_{k}"] = summarize(dn[both, k], got[1][both, k])
        report[f"displacement_{k}"] = summarize((on - (o + d * ln[:, None]))[both, k],
                                                (got[0] - (o + d * got[3][:, None]))[both, k])
    report["path"] = summarize(ln[both], got[3][both])
    with out.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
