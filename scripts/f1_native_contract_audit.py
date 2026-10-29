"""Census initial native construction identities against saved final candidates, without fitting."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from trkfastsim.reco.nativetime import load_times, join_keys, digest
from trkfastsim.reco.nativeref import match_rows

KEYS = ("run", "subrun", "event", "iseed")


def dense_keys(a):
    return np.column_stack([ak.to_numpy(a[k]) for k in KEYS]).astype(np.int64)


def child_keys(parent, values):
    return np.column_stack([np.repeat(parent, ak.to_numpy(ak.num(values)), axis=0),
                            ak.to_numpy(ak.flatten(values)).astype(np.int64)])


def quantiles(a):
    return np.quantile(a, [0, .16, .5, .84, 1]).tolist() if len(a) else []


def audit(check, out):
    gate = json.loads(check.read_text())
    path = gate["inputs"]["baseline"]["path"]
    with uproot.open(path) as f:
        a = f["trkdump/events"].arrays(["run", "subrun", "event", "sd_a0", "kl_d0", "kl_status", "kh_ind", "sd_ind", "kst_sid"])
    one = (ak.num(a.sd_a0) == 1) & (ak.num(a.kl_d0) == 1)
    a = a[one]
    keys = np.column_stack([ak.to_numpy(a[k]) for k in KEYS[:3]] + [np.zeros(len(a), dtype=np.int64)])
    _, meta = load_times(check, path, keys)
    with uproot.open(meta["inputs"]["capture"]["path"]) as f:
        seeds = f["kktrace/seed_contract"].arrays()
        effects = f["kktrace/effects"].arrays([*KEYS, "iteration_ordinal", "kind", "input_index", "strawid", "partner_shindex"])
    s = seeds[join_keys(keys, dense_keys(seeds))]
    initial = child_keys(keys, s.sx_strawid)
    final = child_keys(keys, a.kst_sid)
    unique_final, final_counts = np.unique(final, axis=0, return_counts=True)
    initial_unique, initial_counts = np.unique(initial, axis=0, return_counts=True)
    sx_present = match_rows(initial, unique_final) >= 0
    sid_to_partner = child_keys(keys, s.sh_strawid)
    partner_rows = match_rows(initial, sid_to_partner)
    if np.any(partner_rows < 0):
        raise ValueError("initial crossing lacks matching initial hit straw")
    partners = ak.to_numpy(ak.flatten(s.sh_shindex))[partner_rows]
    captured_partners = ak.to_numpy(ak.flatten(s.sx_partner_shindex))
    partner_disagreement = partners != captured_partners
    initial_hits = child_keys(keys, s.sh_shindex)
    final_hits = child_keys(keys, a.kh_ind)
    sh_present = match_rows(initial_hits, final_hits) >= 0
    lf_hits = child_keys(keys, a.sd_ind)
    seed_present = match_rows(lf_hits, final_hits) >= 0
    native_seed = match_rows(final_hits, initial_hits) >= 0
    inferred_seed = match_rows(final_hits, lf_hits) >= 0
    offset = ak.to_numpy(ak.flatten(s.sx_offset))
    mom = ak.to_numpy(s.parameters)[:, 5]
    predicted = np.repeat(.0125 / (mom / np.hypot(mom, 105.6583755) * 299.792458), ak.to_numpy(ak.num(s.sx_offset)))
    ext = effects[effects.kind == 2]
    ek = np.column_stack([dense_keys(ext), ak.to_numpy(ext.input_index)])
    _, first = np.unique(ek, axis=0, return_index=True)
    ext = ext[first]
    ep = match_rows(dense_keys(ext), dense_keys(seeds))
    idx = ak.to_numpy(ext.input_index)
    is_initial = idx < ak.to_numpy(ak.num(seeds.sx_index))[ep]
    report = {"check": str(check), "check_sha256": digest(check), "inputs": meta["inputs"],
              "n_eligible": len(a), "n_native_initial_hits": len(initial_hits), "n_final_hits": len(final_hits),
              "native_initial_hits_absent_final": int((~sh_present).sum()),
              "linefinder_hits_absent_final": int((~seed_present).sum()),
              "seed_membership_disagreements_final_hits": int((native_seed != inferred_seed).sum()),
              "n_initial_crossings": len(initial), "n_final_crossings": len(final),
              "initial_crossings_absent_final": int((~sx_present).sum()),
              "initial_partner_disagreements_same_straw": int(partner_disagreement.sum()),
              "duplicate_initial_straw_keys": int((initial_counts > 1).sum()),
              "duplicate_final_straw_keys": int((final_counts > 1).sum()),
              "offset_quantiles_ns": quantiles(offset), "offset_minus_wire_radius_over_native_seed_speed_ns": quantiles(offset-predicted),
              "native_seed_momentum_minus_2000_MeV": quantiles(mom-2000),
              "captured_unique_crossings": len(ext), "captured_extension_crossings": int((~is_initial).sum()),
              "extension_offsets_captured": False,
              "limits": ["Final crossing straw identity alone cannot certify native generic identity when duplicates exist",
                         "Extension construction offsets and construction-time references are absent from capture",
                         "Final metadata is not an iteration-local cluster-membership snapshot"]}
    if getattr(audit, "seed_check", False):
        from trkfastsim.reco.nativeseed import load_seed
        x, cov, ranges, seed_meta = load_seed(check, path, keys)
        report["seed_loader"] = {"n": len(x), "momentum_minus_nominal": quantiles(x[:, 5]-2000),
                                  "metadata": seed_meta, "covariance_shape": list(cov.shape)}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with out.open("x") as stream:
        stream.write(text)
    print(text)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed-check", action="store_true")
    a = p.parse_args()
    audit.seed_check = a.seed_check
    audit(a.check, a.out)
