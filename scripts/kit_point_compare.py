"""Compare persisted all-event F2 counts of two tracker points, retaining empty events."""
import argparse
import json
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trkfastsim.analyses.digicmp import _keyid
from trkfastsim.reco.ntuple import TREE


def load(directory):
    p = Path(directory)
    meta = json.loads((p / "metrics.json").read_text())
    prov = json.loads((p / "provenance.json").read_text())
    with uproot.open(p / "EventNtuple.root") as f:
        tree = f[TREE]
        ev = tree["evtinfo"].array()
        keys = np.column_stack([ak.to_numpy(ev[k]) for k in ("run", "subrun", "event")])
        trk = tree["trk"].array()
        n = ak.to_numpy(ak.num(trk["trk.status"]))
        if tree.num_entries != meta["generated_events"] or int(n.sum()) != meta["fit_successes"]:
            raise ValueError("persisted counts disagree with metrics")
    key = _keyid(keys)
    if len(np.unique(key)) != len(key):
        raise ValueError("duplicate event identity")
    order = np.argsort(key)
    return key[order], n[order], meta, prov


def compare(candidate, reference):
    ka, a, ma, pa = load(candidate)
    kb, b, mb, pb = load(reference)
    if not np.array_equal(ka, kb):
        raise ValueError("event populations differ")
    for key in ("inputs",):
        if pa[key] != pb[key]:
            raise ValueError(f"point {key} differ")
    result = {"events": len(ka), "candidate": ma, "reference": mb,
              "configuration": {"candidate": pa["configuration"], "reference": pb["configuration"]},
              "code_differences": {repo: sorted(k for k in set(pa["code"][repo]) | set(pb["code"][repo])
                                    if pa["code"][repo].get(k) != pb["code"][repo].get(k)) for repo in pa["code"]}}
    for name, x, y in (("fits", a, b), ("events_with_fit", a > 0, b > 0)):
        delta = x.astype(float) - y.astype(float)
        mean = np.mean(y)
        if len(delta) < 2 or mean <= 0:
            raise ValueError("paired relative yield requires >=2 events and a nonzero reference yield")
        result[name] = {"candidate": int(x.sum()), "reference": int(y.sum()),
                        "paired_rel_diff": float(delta.mean() / mean),
                        "paired_rel_se": float(delta.std(ddof=1) / np.sqrt(len(delta)) / mean),
                        "both_nonzero": int(np.sum((x > 0) & (y > 0))),
                        "candidate_only": int(np.sum((x > 0) & (y == 0))),
                        "reference_only": int(np.sum((x == 0) & (y > 0)))}
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate")
    p.add_argument("reference")
    args = p.parse_args(argv)
    print(json.dumps(compare(args.candidate, args.reference), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
