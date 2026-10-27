"""Audit the native-time-only arm against saved time-model outputs without refitting."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from trkfastsim.reco import hitreco, kalman_jax as K, nativetime, vr_f1


def reconstructed_time(x, h, c):
    """Same seed midpoint approximation used by the diagnostic time model."""
    x = jnp.r_[x, K.MOM] if x.shape[0] == 5 else x
    valid = h["valid"] & h["inseed"]
    sh = K.straw_poca(x, h["mid"], h["wdir"])[0]
    sx = K.straw_poca(x, c["mid"], c["wdir"])[0] + K.kkmaterial.STRAW["wrad"]
    xv = c["valid"] & (c["hit"] >= 0) & valid[jnp.maximum(c["hit"], 0)]
    times = jnp.r_[jnp.where(valid, x[4] + sh / K.speed(x[5]), jnp.nan),
                   jnp.where(xv, x[4] + sx / K.speed(x[5]), jnp.nan)]
    t = .5 * (jnp.nanmin(times) + jnp.nanmax(times))
    return t, times


def compare_npz(a, b):
    with np.load(a) as left, np.load(b) as right:
        if set(left.files) != set(right.files):
            raise ValueError("saved output schema differs: " + str(a))
        result = {}
        for k in left.files:
            x, y = left[k], right[k]
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError("saved output shape/dtype differs: " + k)
            same = x == y
            if x.dtype.kind in "fc":
                same |= np.isnan(x) & np.isnan(y)
            result[k] = int((~same).sum())
    return {"same_decoded_arrays": not any(result.values()), "different_elements": {k: v for k, v in result.items() if v},
            "n_arrays": len(result), "baseline_sha256": nativetime.digest(a), "candidate_sha256": nativetime.digest(b)}


def run(a):
    out = {"scope": "native immutable time only, compared to preserved stock-inspired time-model arms", "arms": {}}
    for prefix in ("own", "fixed"):
        base, cand = Path(a.baseline_dir) / (prefix + "_stock"), Path(a.candidate_dir) / (prefix + "_stock_native")
        oldmeta, meta = (json.loads((p / "vr2f1.json").read_text()) for p in (base, cand))
        for field in ("input", "own_states", "all_fits", "time_model", "time_legacy", "xing_activity", "env_f1", "material"):
            if oldmeta[field] != meta[field]:
                raise ValueError("non-time configuration changed: " + field)
        if meta["seed_time_provenance"] != "native_ParameterHit_time_only":
            raise ValueError("candidate is not native-time-only")
        x, H, X, ref, _, n = vr_f1.read_inputs(meta["input"], hitreco.load_conditions(a.conditions), "mc",
                                              straw_axis=a.straw_axis, fitok_only=prefix == "fixed")
        keys = np.column_stack([ref["run"], ref["subrun"], ref["event"], np.zeros(n, np.int64)])
        native, _ = nativetime.load_times(a.check, meta["input"], keys)
        with np.load(cand / "native_time.npz") as src:
            np.testing.assert_array_equal(src["keys"], keys)
            np.testing.assert_array_equal(src["constraint_time"], native)
        reconstructed, seed_times = jax.jit(jax.vmap(reconstructed_time))(
            jnp.asarray(x), {k: jnp.asarray(H[k]) for k in ("mid", "wdir", "valid", "inseed")},
            {k: jnp.asarray(X[k]) for k in ("mid", "wdir", "valid", "hit")})
        reconstructed, seed_times = np.asarray(reconstructed), np.asarray(seed_times)
        delta = native - reconstructed
        seed_order_changed = ((seed_times < native[:, None]) != (seed_times < reconstructed[:, None])).any(axis=1)
        result = {"tracks": n, "native_minus_reconstructed_ns_q0_16_50_84_100": np.quantile(delta, [0, .16, .5, .84, 1]).tolist(),
                  "time_values_different": int((delta != 0).sum()), "seed_constraint_order_changed": int(seed_order_changed.sum()),
                  "files": {}}
        for name in ("ids.npz", "validity.npz", "xings.npz") + (("yield.npz",) if prefix == "own" else ()):
            result["files"][name] = compare_npz(base / name, cand / name)
        result["all_saved_arrays_identical"] = all(v["same_decoded_arrays"] for v in result["files"].values())
        out["arms"][prefix] = result
    out["check"] = {"path": a.check, "sha256": nativetime.digest(a.check)}
    with open(a.out, "x") as stream:
        json.dump(out, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("check", "baseline-dir", "candidate-dir", "conditions", "straw-axis", "out"):
        p.add_argument("--" + name, required=True)
    run(p.parse_args())
