"""EventNtuple schema inventory: branch typenames, sub-branches and the awkward types uproot gives a reader.

`inventory(path)` records, for each requested top-level branch of `EventNtuple/ntuple`, the ROOT typename, the
sub-branch typenames (split classes) and the awkward type of `tree[b].array()` and of `tree.arrays([b])` (the two
access forms analysis readers use). The JSON it writes is the stored stock schema the writer tests check against.
"""
import json
import os

import uproot

TREE = "EventNtuple/ntuple"
BRANCHES = ("evtinfo", "evtinfomc", "trk", "trksegs", "trkhits", "trkmats", "trksegpars_kl", "trkmcsim", "trksegsmc",
            "trkhitsmc", "trkcalohit", "trkqual", "trkpid", "trktruth", "trktruthsegs")


def _atype(form_array):
    return str(form_array.type).split(" * ", 1)[1] if " * " in str(form_array.type) else str(form_array.type)


def inventory(path, branches=BRANCHES, tree=TREE, sample_entries=50):
    t = uproot.open(path)[tree]
    out = {"tree": tree, "num_entries": int(t.num_entries), "class": type(t).__name__, "branches": {},
           "source": os.path.basename(path)}
    top = set(t.keys(recursive=False))
    for b in branches:
        if b not in top:
            out["branches"][b] = None
            continue
        br = t[b]
        rec = {"typename": getattr(br, "typename", type(br).__name__), "subbranches": {}}
        for k in br.keys(recursive=True):
            rec["subbranches"][k] = getattr(br[k], "typename", "")
        n = min(sample_entries, t.num_entries)
        arr = br.array(entry_stop=n)
        rec["branch_array_type"] = _atype(arr)
        rec["form"] = arr.layout.form.to_dict()
        rec["arrays_fields"] = list(t.arrays([b], entry_stop=n).fields)
        out["branches"][b] = rec
    out["other_top_branches"] = sorted(top - set(branches))
    return out


def signature(form):
    """Layout-independent structure of an awkward form dict: lists, fixed-size arrays, records (field order kept)
    and primitive dtypes; parameters (ROOT class names) and offset index widths are dropped."""
    c = form["class"]
    if c in ("ListOffsetArray", "ListArray", "ListOffsetArray64", "ListArray64"):
        return ["var", signature(form["content"])]
    if c == "RegularArray":
        return ["reg", form["size"], signature(form["content"])]
    if c == "RecordArray":
        return {"record": [[f, signature(x)] for f, x in zip(form["fields"], form["contents"])]}
    if c == "NumpyArray":
        return form["primitive"] if not form.get("inner_shape") else ["reg", form["inner_shape"], form["primitive"]]
    if c in ("IndexedOptionArray", "UnmaskedArray", "ByteMaskedArray", "BitMaskedArray", "IndexedArray"):
        return ["option", signature(form["content"])]
    raise ValueError(f"unexpected form class {c}")


def leaves(sig, prefix=""):
    """(dotted path, primitive, list depth) of every leaf of a signature."""
    if isinstance(sig, str):
        return [(prefix, sig, 0)]
    if isinstance(sig, dict):
        out = []
        for f, s in sig["record"]:
            out += leaves(s, f"{prefix}.{f}" if prefix else f)
        return out
    if sig[0] == "var":
        return [(p, t, d + 1) for p, t, d in leaves(sig[1], prefix)]
    if sig[0] == "reg":
        inner = sig[2]
        return [(p, f"{t}[{sig[1]}]", d) for p, t, d in leaves(inner, prefix)]
    return leaves(sig[1], prefix)


def fields_of(schema, branch):
    """Leaf field names (dotted, without the branch prefix) of one branch in a stored schema."""
    rec = schema["branches"][branch]
    return sorted(k.split(".", 1)[1] if k.startswith(branch + ".") else k for k in rec["subbranches"])


def main(path, out, branches=BRANCHES):
    inv = inventory(path, branches)
    with open(out, "w") as f:
        json.dump(inv, f, indent=1, sort_keys=True)
    print(json.dumps({b: (None if v is None else {"typename": v["typename"], "n_sub": len(v["subbranches"])})
                      for b, v in inv["branches"].items()}, indent=1))
