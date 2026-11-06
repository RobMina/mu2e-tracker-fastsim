"""Opt-in measured-hit resident wire-helix reconstruction; not CentralHelix.

Only the explicit host preparation helper in residentio performs host packing.
This module composes under one outer jax.jit and retains original hit identities.
"""
from functools import partial

import jax
import jax.numpy as jnp

from .helixseed_jax import fit_helix_seed
from .helixfit_jax import fit_wire_helix

jax.config.update("jax_enable_x64", True)

GROUP_STATUS = {"padding": 0, "converged": 1, "insufficient_hits": 2,
                "wide_time": 3, "hit_capacity": 4, "invalid_input": 5,
                "duplicate_straw": 6, "seed_failed": 7, "fit_failed": 8}
HIT_STATUS = {"padding": 0, "grouped": 1, "flag_rejected": 2, "invalid_input": 3}
BAD_FLAGS = (1 << 6) | (1 << 23) | (1 << 24)
TDIV = 1 << 10
HIT_FIELDS = ("ev", "sid", "time", "flag", "wdist", "uvar", "mid", "direction", "hlen", "valid")


def _check(keys, h, max_hits, group_batch, max_iterations):
    if keys.ndim != 2 or keys.shape[1] != 3 or keys.dtype.kind not in "iu":
        raise ValueError("event_keys must be integer [events,3]")
    if keys.dtype.itemsize > 8 or keys.dtype.kind == "u" and keys.dtype.itemsize == 8:
        raise ValueError("event_keys must fit signed int64; use the checked host adapter")
    for name, value, low, high in (("max_hits", max_hits, 4, 512),
                                  ("group_batch", group_batch, 1, 2048),
                                  ("max_iterations", max_iterations, 0, 120)):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low},{high}]")
    if group_batch * max_hits > 262144:
        raise ValueError("group_batch * max_hits exceeds workspace cell bound")
    if set(h) != set(HIT_FIELDS):
        raise ValueError(f"hits must have exactly {HIT_FIELDS}")
    n = h["ev"].shape[0]
    for k in HIT_FIELDS:
        shape = (n, 3) if k in ("mid", "direction") else (n,)
        kind = "b" if k == "valid" else "iu" if k in ("ev", "sid", "flag") else "fiu"
        if h[k].shape != shape or h[k].dtype.kind not in kind:
            raise ValueError(f"invalid shape or dtype for hits[{k}]")
        if h[k].dtype.kind == "u" and h[k].dtype.itemsize == 8:
            raise ValueError(f"hits[{k}] must fit signed int64")
    return n, keys.shape[0]


def _empty(keys, h):
    n, ne = h["ev"].shape[0], keys.shape[0]
    out = dict(event_keys=keys, n_groups=jnp.int64(0), batch_valid=jnp.bool_(True),
               event_n_hits=jnp.zeros(ne, jnp.int64), event_n_groups=jnp.zeros(ne, jnp.int64),
               event_n_converged=jnp.zeros(ne, jnp.int64), event_invalid=jnp.zeros(ne, bool),
               hit_group=jnp.full(n, -1, jnp.int64), hit_status=jnp.zeros(n, jnp.int32),
               group_valid=jnp.zeros(n, bool), group_ev=jnp.full(n, -1, jnp.int64),
               group_n_hits=jnp.zeros(n, jnp.int64), group_time_span=jnp.zeros(n),
               group_status=jnp.zeros(n, jnp.int32), group_seed_status=jnp.zeros(n, jnp.int32),
               group_fit_status=jnp.zeros(n, jnp.int32), group_success=jnp.zeros(n, bool),
               group_seed_attempted=jnp.zeros(n, bool), group_fit_attempted=jnp.zeros(n, bool),
               group_condition=jnp.full(n, jnp.inf), group_rank=jnp.zeros(n, jnp.int32),
               group_gradient_norm=jnp.zeros(n),
               group_local_parameters=jnp.full((n, 5), jnp.nan), group_z_ref=jnp.full(n, jnp.nan),
               group_objective_valid=jnp.zeros(n, bool),
               group_initial_objective=jnp.zeros(n), group_final_objective=jnp.zeros(n),
               group_iterations=jnp.zeros(n, jnp.int32), group_accepted_steps=jnp.zeros(n, jnp.int32),
               hit_doca=jnp.zeros(n), hit_wire_pull=jnp.zeros(n), hit_poca_valid=jnp.zeros(n, bool))
    out.update({"input_" + k: v for k, v in h.items()})
    return out


@partial(jax.jit, static_argnames=("max_hits", "group_batch", "max_iterations"))
def reconstruct(event_keys, hits, *, bz, max_hits=128, group_batch=32, max_iterations=40):
    """Device arrays in/out, one geometric hypothesis per measured time group.

    Keys are lossless [E,3]; hits are flat, padded only where valid=False.
    Required hit fields are HIT_FIELDS. Wire direction is full 3D; mid/length,
    wdist and uvar use mm/mm2. No truth or track memberships are accepted.
    Time grouping uses <=20 ns adjacent gaps, with a <=100 ns group-span gate.
    All group memberships survive; over-capacity groups are NOT partially fit.
    max_hits/group_batch bound fit workspace, not total input/output storage.
    Invalid selected timestamps invalidate their whole event. Bad geometry
    invalidates its group. Duplicate straw pulses are explicitly unsupported.
    Empty events and every input row survive. Status=converged means only
    numerical convergence of the wire objective, never physical track validity.
    """
    n, ne = _check(event_keys, hits, max_hits, group_batch, max_iterations)
    out = _empty(event_keys, hits)
    bz = jnp.asarray(bz)
    if bz.shape != () or bz.dtype.kind not in "fiu":
        raise ValueError("bz must be a real numeric scalar")
    bz = bz.astype(jnp.float64)
    field_ok = jnp.isfinite(bz) & (bz != 0)
    order_keys = jnp.lexsort((event_keys[:, 2], event_keys[:, 1], event_keys[:, 0]))
    sk = event_keys[order_keys]
    keys_ok = jnp.all(event_keys >= 0) & ~jnp.any(jnp.all(sk[1:] == sk[:-1], axis=1))
    if n == 0 or ne == 0:
        bad = jnp.any(hits["valid"]) if ne == 0 else jnp.bool_(False)
        out["batch_valid"] = keys_ok & field_ok & ~bad
        out["event_invalid"] = jnp.full(ne, ~keys_ok | ~field_ok)
        out["hit_status"] = jnp.where(hits["valid"], HIT_STATUS["invalid_input"], 0)
        return out
    ev, sid, flag = (hits[k].astype(jnp.int64) for k in ("ev", "sid", "flag"))
    present = hits["valid"]
    ev_ok = (ev >= 0) & (ev < ne)
    e = jnp.clip(ev, 0, ne - 1)
    sid_ok = (sid >= 0) & ((sid >> 10) < 36) & ((sid & 127) < 96) & (((sid >> 7) & 7) < 6)
    identity_ok = ev_ok & sid_ok & (flag >= 0) & (flag <= 2**32 - 1)
    selected = ((flag & BAD_FLAGS) == 0) & ((flag & TDIV) != 0)
    time_ok = jnp.isfinite(hits["time"])
    bad = present & (~identity_ok | (selected & ~time_ok))
    global_bad = ~keys_ok | ~field_ok | jnp.any(present & ~ev_ok)
    event_bad = jnp.zeros(ne, jnp.int32).at[e].max((bad & ev_ok).astype(jnp.int32)) > 0
    event_bad |= global_bad
    member = present & identity_ok & selected & time_ok
    order = jnp.lexsort((jnp.arange(n), sid, jnp.where(member, hits["time"], jnp.inf),
                        jnp.where(member, ev, ne)))
    m, se, t = member[order], ev[order], hits["time"][order].astype(jnp.float64)
    begin = m & jnp.concatenate((jnp.ones(1, bool), (se[1:] != se[:-1]) | ((t[1:] - t[:-1]) > 20.)))
    sg = jnp.cumsum(begin, dtype=jnp.int64) - 1
    ng = jnp.sum(begin, dtype=jnp.int64)
    g = jnp.clip(sg, 0, n - 1)
    count = jnp.zeros(n, jnp.int64).at[g].add(m.astype(jnp.int64))
    start = jnp.full(n, n, jnp.int64).at[g].min(jnp.where(begin, jnp.arange(n), n))
    end = jnp.zeros(n, jnp.int64).at[g].max(jnp.where(m, jnp.arange(n), 0))
    first = jnp.clip(start, 0, n - 1)
    live = jnp.arange(n) < ng
    ge = jnp.where(live, se[first], -1)
    span = jnp.where(live, t[end] - t[first], 0.)
    hit_group = jnp.full(n, -1, jnp.int64).at[order].set(jnp.where(m, sg, -1))
    ds = jnp.max(jnp.abs(hits["direction"]), axis=1)
    geom_ok = (jnp.all(jnp.isfinite(hits["mid"]), axis=1)
               & jnp.all(jnp.isfinite(hits["direction"]), axis=1) & (ds > 0)
               & jnp.isfinite(hits["wdist"]) & jnp.isfinite(hits["uvar"]) & (hits["uvar"] > 0)
               & jnp.isfinite(hits["hlen"]) & (hits["hlen"] > 0))
    gbad = jnp.zeros(n, jnp.int32).at[g].max((m & ~geom_ok[order]).astype(jnp.int32)) > 0
    duporder = jnp.lexsort((sid, jnp.where(member, hit_group, n)))
    dg, di = hit_group[duporder], sid[duporder]
    duplicate = (dg[1:] >= 0) & (dg[1:] == dg[:-1]) & (di[1:] == di[:-1])
    duplicates = jnp.zeros(n, jnp.int32).at[jnp.clip(dg[1:], 0, n - 1)].max(duplicate.astype(jnp.int32)) > 0
    status = jnp.full(n, GROUP_STATUS["seed_failed"], jnp.int32)
    for predicate, label in ((count < 4, "insufficient_hits"), (duplicates, "duplicate_straw"),
                             (span > 100., "wide_time"), (count > max_hits, "hit_capacity"),
                             (gbad | event_bad[jnp.clip(ge, 0, ne - 1)], "invalid_input")):
        status = jnp.where(predicate, GROUP_STATUS[label], status)
    status = jnp.where(live, status, 0)
    out.update(n_groups=ng, batch_valid=~global_bad & ~jnp.any(bad), event_invalid=event_bad,
               event_n_hits=jnp.zeros(ne, jnp.int64).at[e].add((present & ev_ok).astype(jnp.int64)),
               event_n_groups=jnp.zeros(ne, jnp.int64).at[e[order]].add(begin.astype(jnp.int64)),
               hit_group=hit_group, hit_status=jnp.where(~present, 0, jnp.where(bad, 3, jnp.where(member, 1, 2))),
               group_valid=live, group_ev=ge, group_n_hits=count, group_time_span=span, group_status=status)

    def step(carry):
        offset, data = carry
        rows = offset + jnp.arange(group_batch)
        r = jnp.clip(rows, 0, n - 1)
        exists = rows < ng
        eligible = exists & (status[r] == GROUP_STATUS["seed_failed"])
        columns = jnp.arange(max_hits)[None, :]
        in_group = exists[:, None] & (columns < count[r, None])
        ix = order[jnp.clip(start[r, None] + columns, 0, n - 1)]
        mask = in_group & eligible[:, None]
        mid = hits["mid"][ix].astype(jnp.float64)
        direction = hits["direction"][ix].astype(jnp.float64)
        safe_dir = jnp.where(mask[..., None], direction, jnp.array([1., 0., 0.]))
        normed = safe_dir / jnp.max(jnp.abs(safe_dir), axis=-1, keepdims=True)
        normed /= jnp.linalg.norm(normed, axis=-1, keepdims=True)
        u = hits["wdist"][ix].astype(jnp.float64)
        positions = jnp.where(mask[..., None], mid + u[..., None] * normed, 0.)
        seed = fit_helix_seed(positions, mask, bz=bz)
        radius, pitch, phase = (seed[k] for k in ("radius", "pitch", "phase_ref"))
        safe_pitch = jnp.where(seed["success"], pitch, 1.)
        p = jnp.stack((seed["center_xy"][:, 0] + radius * jnp.cos(phase),
                       seed["center_xy"][:, 1] + radius * jnp.sin(phase),
                       -radius / safe_pitch * jnp.sin(phase), radius / safe_pitch * jnp.cos(phase),
                       1. / safe_pitch), axis=1)
        fit = fit_wire_helix(p, seed["z_ref"], mid, direction, u, hits["uvar"][ix], hits["hlen"][ix],
                             mask, seed_valid=seed["success"] & eligible, max_iterations=max_iterations)
        passed = eligible & seed["success"] & fit["success"]
        gs = jnp.where(seed["success"], jnp.where(passed, 1, 8), 7)
        updates = {"group_status": jnp.where(eligible, gs, status[r]),
                   "group_seed_status": jnp.where(eligible, seed["status"], 0),
                   "group_fit_status": jnp.where(eligible & seed["success"], fit["status"], 0),
                   "group_success": passed, "group_seed_attempted": eligible,
                   "group_fit_attempted": eligible & seed["success"]}
        for name in ("local_parameters", "z_ref", "objective_valid", "initial_objective", "final_objective",
                     "iterations", "accepted_steps", "condition", "rank", "gradient_norm"):
            updates["group_" + name] = fit[name]
        for key, value in updates.items():
            data[key] = data[key].at[jnp.where(exists, rows, n)].set(value, mode="drop")
        for key, value in (("hit_doca", fit["doca"]), ("hit_wire_pull", fit["wire_pull"]),
                           ("hit_poca_valid", fit["poca_valid"] & fit["poca_converged"] & fit["poca_inside"])):
            dest = jnp.where(mask, ix, n)
            data[key] = data[key].at[dest].set(value, mode="drop")
        return offset + group_batch, data

    _, out = jax.lax.while_loop(lambda carry: carry[0] < ng, step, (jnp.int64(0), out))
    out["event_n_converged"] = jnp.zeros(ne, jnp.int64).at[jnp.clip(ge, 0, ne - 1)].add(
        out["group_success"].astype(jnp.int64))
    return out
