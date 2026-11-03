"""yieldpairs -- where a paired KKLine yield difference comes from (diagnostic).

Two arms on the same generated events and the same selection (e.g. two step options, same steps seed): per selected
event the number of good tracks (status > 0), of failed fits (status <= 0) and of digis in each arm. Reports the track
transitions (events losing / gaining a good track), whether the arm without the track has a failed fit (the fit or its
hit selection rejected it) or no fit at all (no time cluster or seed), and the loss and gain rates per digi difference.

    yieldpairs CAND_NTUPLES... --compare REF_NTUPLES... --steps CAND_STEPS... --steps-ref REF_STEPS...
               --hits CAND_HITS... --hits-ref REF_HITS... --outdir DIR

Output: <outdir>/yieldpairs.json.
"""
import json
import os

import awkward as ak
import numpy as np
import uproot

from trkfastsim import tables as T
from trkfastsim.validation import tv2compare as TV

DDIGI_EDGES = np.array([-1e9, -0.5, 0.5, 2.5, 5.5, 1e9])     # digi difference cand - ref: < 0, 0, 1-2, 3-5, >= 6
DDIGI_LABELS = ["< 0", "0", "1-2", "3-5", ">= 6"]


def _tracks(paths, sel):
    """Per selected event (sorted keys): good tracks (status > 0) and failed fits (status <= 0)."""
    keys, good, bad = [], [], []
    for path in T.expand(paths):
        a = uproot.open(path)[TV.TREE].arrays(["evtinfo/run", "evtinfo/subrun", "evtinfo/event", "trk/trk.status"],
                                              library="ak")
        k = TV._keyid_rows(np.stack([ak.to_numpy(a[f"evtinfo/{c}"]) for c in ("run", "subrun", "event")], 1))
        m = np.isin(k, sel)
        st = a["trk/trk.status"][m]
        keys.append(k[m])
        good.append(ak.to_numpy(ak.sum(st > 0, axis=1)))
        bad.append(ak.to_numpy(ak.sum(st <= 0, axis=1)))
    k, g, b = np.concatenate(keys), np.concatenate(good), np.concatenate(bad)
    o = np.argsort(k)
    if not np.array_equal(k[o], sel):
        raise ValueError("ntuple events differ from the selected events")
    return g[o], b[o]


def _digis(paths, sel):
    h = TV._hits(paths, sel)
    o = np.argsort(h["key"])
    if not np.array_equal(h["key"][o], sel):
        raise ValueError("hit-table events differ from the selected events")
    return h["ndigi"][o].astype(np.int64)


def run(inputs, outdir, compare=None, steps=None, steps_ref=None, hits=None, hits_ref=None, **kw):
    os.makedirs(outdir, exist_ok=True)
    sel_c, _ = TV._selected(steps)
    sel_r, _ = TV._selected(steps_ref)
    if not np.array_equal(sel_c, sel_r):
        raise SystemExit("the arms select different events: yieldpairs needs the same selection")
    gc, bc = _tracks(inputs, sel_c)
    gr, br = _tracks(compare, sel_r)
    dd = _digis(hits, sel_c) - _digis(hits_ref, sel_r)
    lost, gained = gc < gr, gc > gr
    db = np.digitize(dd, DDIGI_EDGES) - 1
    by = {}
    for j, lab in enumerate(DDIGI_LABELS):
        m = db == j
        n = int(m.sum())
        by[lab] = {"events": n, "lost": int(lost[m].sum()), "gained": int(gained[m].sum()),
                   "net_tracks": int((gc - gr)[m].sum()),
                   "net_per_1000_events": float(1000 * (gc - gr)[m].sum() / max(n, 1))}
    res = {"events": int(len(sel_c)), "tracks": [int(gc.sum()), int(gr.sum())],
           "net_tracks": int((gc - gr).sum()), "events_lost": int(lost.sum()), "events_gained": int(gained.sum()),
           "lost_with_failed_fit": int((lost & (bc > br)).sum()), "lost_without_fit": int((lost & (gc + bc < gr + br)).sum()),
           "gained_with_failed_fit_in_ref": int((gained & (br > bc)).sum()),
           "failed_fits": [int(bc.sum()), int(br.sum())],
           "digi_difference_mean": float(dd.mean()), "events_with_more_digis": int((dd > 0).sum()),
           "by_digi_difference": by,
           "inputs": {"cand": [os.path.basename(p) for p in T.expand(inputs)],
                      "ref": [os.path.basename(p) for p in T.expand(compare)]}}
    with open(os.path.join(outdir, "yieldpairs.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "inputs"}, indent=1))
    return res
