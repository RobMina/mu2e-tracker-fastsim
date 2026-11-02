"""Paired per-shard comparison of chain_bench reports: arm B against reference arm A on the same generator tables,
and a null arm C (A's backends with another seed) against A as the floor.

For a count c, d_s = c_B,s - c_A,s over the S shared shards; the relative difference sum_s d_s / sum_s c_A,s has
the standard error sqrt(S/(S-1) sum_s (d_s - mean d)^2) / sum_s c_A,s. For a histogram, the same per bin over the
shard sums: chi2 = sum_b D_b^2 / var_b with D_b = sum_s d_sb and var_b = S/(S-1) sum_s (d_sb - mean_b)^2, over bins
with var_b > 0 and at least `min_count` entries in A. With a variance estimated from S shards, a correct arm has
<chi2/ndf> = (S-1)/(S-3), not 1; the null arm's value is the floor and the flag is above max(2, 3 x floor).
"""
import argparse
import json
from pathlib import Path

import numpy as np


def _shards(report):
    r = json.loads(Path(report).read_text())
    return r, {s["gen"]: s for s in r["shards"]}


def paired_counts(a, b, names):
    S = len(a)
    out = {}
    for k in names:
        ca = np.array([s["counts"][k] for s in a], float)
        cb = np.array([s["counts"][k] for s in b], float)
        d = cb - ca
        tot = ca.sum()
        se = np.sqrt(S / (S - 1) * ((d - d.mean()) ** 2).sum()) if S > 1 else float("nan")
        out[k] = {"A": int(ca.sum()), "B": int(cb.sum()), "rel": float(d.sum() / tot) if tot else 0.0,
                  "rel_se": float(se / tot) if tot else 0.0, "identical_shards": int((d == 0).sum())}
        out[k]["pull"] = out[k]["rel"] / out[k]["rel_se"] if out[k]["rel_se"] > 0 else 0.0
    return out


def paired_hist(a, b, name, min_count=20):
    ha = np.array([s["histograms"][name] for s in a], float)
    hb = np.array([s["histograms"][name] for s in b], float)
    S = len(a)
    d = hb - ha
    D = d.sum(0)
    var = S / (S - 1) * ((d - d.mean(0)) ** 2).sum(0)
    m = (var > 0) & (ha.sum(0) >= min_count)
    ndf = int(m.sum())
    chi2 = float((D[m] ** 2 / var[m]).sum())
    return {"chi2": chi2, "ndf": ndf, "chi2_ndf": chi2 / ndf if ndf else 0.0,
            "expected_chi2_ndf": (S - 1) / (S - 3) if S > 3 else float("nan"),
            "identical": bool(np.array_equal(ha, hb))}


def compare(ref, cand, null=None, min_count=20):
    ra, A = _shards(ref)
    rb, B = _shards(cand)
    common = sorted(set(A) & set(B))
    if null:
        rc, C = _shards(null)
        common = sorted(set(common) & set(C))
    if len(common) < 4:
        raise ValueError(f"need at least 4 shared shards, have {len(common)}")
    a, b = [A[g] for g in common], [B[g] for g in common]
    for x, y in zip(a, b):
        if x["counts"]["generated_events"] != y["counts"]["generated_events"]:
            raise ValueError(f"{x['gen']}: generated events differ between arms")
    names = list(a[0]["counts"])
    hists = list(a[0]["histograms"])
    out = {"shards": common, "S": len(common), "ref": {"backends": ra["backends"], "seed": ra["seed"]},
           "cand": {"backends": rb["backends"], "seed": rb["seed"]},
           "counts": paired_counts(a, b, names), "hists": {h: paired_hist(a, b, h, min_count) for h in hists},
           "stage_hash_identical": {k: sum(x["hashes"][k] == y["hashes"][k] for x, y in zip(a, b))
                                    for k in a[0]["hashes"]}}
    if null:
        c = [C[g] for g in common]
        out["null"] = {"backends": rc["backends"], "seed": rc["seed"], "counts": paired_counts(a, c, names),
                       "hists": {h: paired_hist(a, c, h, min_count) for h in hists}}
        out["flags"] = sorted(h for h in hists if out["hists"][h]["chi2_ndf"] >
                              max(2.0, 3.0 * out["null"]["hists"][h]["chi2_ndf"]))
    return out


def compare_products(a, b):
    """Per-array comparison of two chain_bench --save-steps files: identical, or the row-count difference and the
    largest absolute difference of a float column."""
    za, zb = np.load(a), np.load(b)
    out = {}
    for k in sorted(set(za.files) | set(zb.files)):
        if k not in za.files or k not in zb.files:
            out[k] = "missing"
            continue
        x, y = za[k], zb[k]
        if x.shape == y.shape and np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
            out[k] = "identical"
        elif x.shape != y.shape:
            out[k] = f"rows {x.shape[0]} vs {y.shape[0]}"
        else:
            diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
            out[k] = f"differ in {int((diff > 0).sum())} of {diff.size}, max |d| {float(diff.max()):.3g}"
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ref", help="reference chain_bench report.json (arm A)")
    p.add_argument("cand", help="candidate report.json (arm B)")
    p.add_argument("--null", help="null report.json (arm A's backends, another seed)")
    p.add_argument("--min-count", type=int, default=20)
    p.add_argument("--out", type=Path)
    p.add_argument("--products", action="store_true", help="ref and cand are --save-steps .npz files")
    a = p.parse_args(argv)
    res = compare_products(a.ref, a.cand) if a.products else compare(a.ref, a.cand, a.null, a.min_count)
    text = json.dumps(res, indent=1, allow_nan=False)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
