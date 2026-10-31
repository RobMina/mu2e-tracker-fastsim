"""Opt-in ordered stored-material transport used by the NumPy step generator."""
import numpy as np
import pandas as pd

from mu2efastsim import eloss, materials, transport as tr
from . import crossings, geometry, ledger, storednavigation
from .flighttime import linear_kinetic_dt_ns

MODE = "ordered-stored-mean-v1"
SAMPLED_MODE = "ordered-stored-sampled-v1"
MODES = (MODE, SAMPLED_MODE)


def validate_provider_partition(partition, lengths):
    """Require a complete disjoint physical-material partition from an explicit provider."""
    from . import boundarypolicy
    lengths = np.asarray(lengths, float)
    boundarypolicy._validate(partition.assign(sign=1), lengths)
    if "kind" not in partition or not partition.kind.isin(["vacuum", "support", "thin"]).all():
        raise ValueError("provider requires explicit vacuum/support/thin kinds")
    if ((partition.material == "vacuum") != (partition.kind == "vacuum")).any():
        raise ValueError("provider material and kind disagree")
    ordered = partition.sort_values(["lane", "s0_mm"], kind="stable")
    if (ordered.s1_mm <= ordered.s0_mm).any():
        raise ValueError("provider partition requires positive intervals")
    first = ordered.drop_duplicates("lane", keep="first")
    last = ordered.drop_duplicates("lane", keep="last")
    if (not np.array_equal(first.lane.to_numpy(), np.flatnonzero(lengths > 0)) or
            (first.s0_mm != 0).any() or not np.array_equal(last.s1_mm.to_numpy(), lengths[lengths > 0])):
        raise ValueError("provider partition must cover every requested lane exactly")
    previous = ordered.groupby("lane").s1_mm.shift()
    if (ordered.loc[previous.notna(), "s0_mm"] != previous[previous.notna()]).any():
        raise ValueError("provider partition contains a gap or overlap")


def transport(interior, rng, origin, direction, momentum, geom, *, tail, tail_k,
              max_path_mm=4000., max_step_mm=10., fractional_loss=.05,
              stop_ke_MeV=.5, max_iter=20000, boundary_resolution_mm=1e-9,
              max_boundary_relabel_mm=1e-7, progress=None, signed_endpoint_budget_mm=None, mode=MODE,
              stopping_evaluation="entry", navigation_provider=None, forward_plane_y_mm=None, material_resolver=None, bounded_forward_lookahead=False):
    """Advance one material slice per live lane, then re-navigate after scattering.

    Mean mode chooses deterministic unrestricted loss; sampled mode uses Urban
    proposals and gas response-coupled debit. Both use linear KE in each slice.
    Stops at the positive model cutoff are absorbing. Stored
    same-medium support is unioned, other overlaps refused. Angular Highland/tail
    kicks occur at material slice ends; there is no transverse displacement.
    Gas response is a restricted sample, capped by the mean only in mean mode;
    its complement is unassigned transfer, not demonstrated escape. Libraries
    are unsupported. The finite requested path is a bound, not a GDML world exit.
    Explicit midpoint evaluation uses a half-mean-loss predictor for both stopping
    power and proposals; entry evaluation remains the default in both modes.
    A separately named navigation_provider is opt-in conditional sensitivity only;
    it must return complete support/thin/vacuum partitions on every fresh ray.
    """
    from .steps import STRAW_GAS, COMPONENTS, _group_tail, _kink, _kink_core
    if mode not in MODES:
        raise ValueError("explicit supported ordered mode required")
    sampled_loss = mode == SAMPLED_MODE
    if navigation_provider is not None and (not callable(navigation_provider) or
            not isinstance(getattr(navigation_provider, "policy", None), str) or not navigation_provider.policy):
        raise ValueError("named callable navigation provider required")
    navigate = storednavigation.intervals if navigation_provider is None else navigation_provider
    if material_resolver is not None and (not callable(material_resolver) or not isinstance(getattr(material_resolver, "policy", None), str)):
        raise ValueError("explicit named callable material resolver required")
    resolve_material = materials.get if material_resolver is None else material_resolver
    if forward_plane_y_mm is not None and not np.isfinite(forward_plane_y_mm):
        raise ValueError("finite forward-domain plane required")
    if type(bounded_forward_lookahead) is not bool or (bounded_forward_lookahead and forward_plane_y_mm is None):
        raise ValueError("bounded lookahead requires explicit forward plane")
    if stopping_evaluation not in ("entry", "midpoint"):
        raise ValueError("stopping evaluation must be entry or midpoint")
    for value in (max_path_mm, max_step_mm, fractional_loss, stop_ke_MeV):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("ordered path settings must be finite positive")
    for value in (boundary_resolution_mm, max_boundary_relabel_mm):
        if not np.isfinite(value) or value < 0:
            raise ValueError("boundary resolutions must be finite nonnegative")
    if fractional_loss > 1:
        raise ValueError("fractional loss must not exceed one")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
        raise ValueError("positive integer iteration cap required")
    o, d = np.array(origin, float, copy=True), np.array(direction, float, copy=True)
    momentum = np.asarray(momentum, float)
    if (momentum.ndim != 1 or o.shape != (len(momentum), 3) or d.shape != o.shape
            or not all(np.isfinite(v).all() for v in (o, d, momentum)) or (momentum < 0).any()
            or not np.allclose(np.linalg.norm(d, axis=1), 1., rtol=0, atol=1e-10)):
        raise ValueError("finite (N,3) origins/unit directions and nonnegative (N,) momenta required")
    interior.last_forward_refusal = None
    initial = ledger.kinetic(momentum)
    ke = initial.copy()
    n = len(ke)
    distance, clock, xcum = np.zeros(n), np.zeros(n), np.zeros(n)
    anchor, nav_s = o.copy(), np.zeros(n)
    boundary_relabel = np.zeros(n)
    query_end = np.zeros(n)
    lookahead_records = []
    comp = {k: np.zeros(n) for k in COMPONENTS}
    stopped = ke <= stop_ke_MeV
    rows, slices, endpoint_records, leading_records = [], [], [], []
    dirty = np.ones(n, bool)
    cached = pd.DataFrame(columns=["lane", "s0_mm", "s1_mm", "material", "kind"])
    stats = dict(kinks=0, tail_scatters=0, stopped=int(stopped.sum()), iterations=0, capped=0,
                 navigation=storednavigation.POLICY if navigation_provider is None else navigation_provider.policy,
                 mode=mode, gas_capped_slices=0,
                 gas_sampled_MeV=0., gas_applied_MeV=0., stopping_evaluation=stopping_evaluation)
    for iteration in range(max_iter):
        active = np.flatnonzero(~stopped & (distance < max_path_mm))
        if not len(active):
            break
        exhausted = active[(query_end[active] > 0) & (nav_s[active] >= query_end[active])]
        if bounded_forward_lookahead and len(exhausted):
            anchor[exhausted], nav_s[exhausted], dirty[exhausted] = o[exhausted], 0., True
        fresh = active[dirty[active]]
        if len(fresh):
            def endpoints(evidence):
                changed = evidence["endpoints"]
                changed = changed[changed.shift_mm != 0].copy()
                changed["lane"] = fresh[changed.lane.to_numpy(dtype=np.int64)]
                changed["iteration"] = iteration
                endpoint_records.append(changed)
                leading = evidence.get("leading_sources")
                if leading is not None and len(leading):
                    leading = leading.copy()
                    leading["lane"] = fresh[leading.lane.to_numpy(dtype=np.int64)]
                    leading["iteration"] = iteration
                    np.add.at(boundary_relabel, leading.lane.to_numpy(), (leading.s1_mm - leading.s0_mm).to_numpy())
                    if (boundary_relabel > max_boundary_relabel_mm).any():
                        raise RuntimeError("ordered boundary relabel budget exceeded")
                    leading_records.append(leading)
            query_length = max_path_mm - distance[fresh] + nav_s[fresh]
            if bounded_forward_lookahead:
                upward = d[fresh, 1] > 0
                crossing = np.divide(forward_plane_y_mm - anchor[fresh, 1], d[fresh, 1],
                    out=np.full(len(fresh), np.inf), where=upward)
                reduce = upward & (crossing > max_step_mm) & (crossing < query_length)
                bounded = np.minimum(query_length, np.minimum(100., np.maximum(max_step_mm, .5 * crossing)))
                query_length = np.where(reduce, bounded, query_length)
                if reduce.any():
                    lookahead_records.append(pd.DataFrame(dict(lane=fresh[reduce], iteration=iteration,
                        full_requested_mm=(max_path_mm - distance[fresh] + nav_s[fresh])[reduce],
                        query_length_mm=query_length[reduce], crossing_mm=crossing[reduce],
                        x=anchor[fresh[reduce], 0], y=anchor[fresh[reduce], 1], z=anchor[fresh[reduce], 2],
                        ux=d[fresh[reduce], 0], uy=d[fresh[reduce], 1], uz=d[fresh[reduce], 2])))
            try:
                if forward_plane_y_mm is not None:
                    from .orderedhandoff import validate_forward_query
                    try:
                        validate_forward_query(anchor[fresh], d[fresh], query_length, forward_plane_y_mm)
                    except ValueError:
                        interior.last_forward_refusal = dict(interior_lanes=fresh.copy(), iteration=iteration,
                            origin=anchor[fresh].copy(), direction=d[fresh].copy(),
                            lengths=query_length.copy(), plane_y_mm=forward_plane_y_mm)
                        raise
                if navigation_provider is not None and hasattr(navigation_provider, "bind_query_lanes"):
                    navigation_provider.bind_query_lanes(fresh.copy(), iteration)
                updated = navigate(interior.shells, geom, anchor[fresh], d[fresh],
                    query_length, endpoint_budget_mm=signed_endpoint_budget_mm,
                    endpoint_evidence=endpoints if signed_endpoint_budget_mm is not None else None,
                    leading_resolution_mm=boundary_resolution_mm if signed_endpoint_budget_mm is not None else None)
                if navigation_provider is not None:
                    validate_provider_partition(updated, query_length)
            except ValueError as error:
                details = getattr(error, "details", None)
                if details is not None and "lane" in details:
                    error.details = dict(details, interior_lane=int(fresh[details["lane"]]), iteration=iteration)
                raise
            query_end[fresh] = query_length
            updated = updated.copy()
            updated["lane"] = fresh[updated.lane.to_numpy()]
            retained = cached[~cached.lane.isin(fresh)]
            cached = updated if retained.empty else pd.concat([retained, updated], ignore_index=True)
            dirty[fresh] = False
        partition = cached[cached.lane.isin(active)].sort_values(["lane", "s0_mm"], kind="stable").copy()
        cursor = nav_s[partition.lane.to_numpy(dtype=np.int64)]
        partition = partition[partition.s1_mm.to_numpy() > cursor].copy()
        partition["s0_mm"] = np.maximum(partition.s0_mm.to_numpy(), nav_s[partition.lane.to_numpy(dtype=np.int64)])
        partition["lane"] = np.searchsorted(active, partition.lane.to_numpy())
        tiny = (partition.s1_mm - partition.s0_mm).to_numpy() <= boundary_resolution_mm
        if tiny.any():
            for _ in range(32):
                first_rows = ~partition.lane.duplicated()
                micro = first_rows.to_numpy() & ((partition.s1_mm - partition.s0_mm).to_numpy() <= boundary_resolution_mm)
                next_same = np.r_[partition.lane.to_numpy()[1:] == partition.lane.to_numpy()[:-1], False]
                merge = np.flatnonzero(micro & next_same)
                if not len(merge):
                    break
                lanes = active[partition.lane.to_numpy()[merge]]
                boundary_relabel[lanes] += (partition.s1_mm - partition.s0_mm).to_numpy()[merge]
                if (boundary_relabel > max_boundary_relabel_mm).any():
                    raise RuntimeError("ordered boundary relabel budget exceeded")
                partition.iloc[merge + 1, partition.columns.get_loc("s0_mm")] = partition.s0_mm.to_numpy()[merge]
                partition = partition.drop(partition.index[merge]).reset_index(drop=True)
            else:
                raise RuntimeError("ordered boundary resolution chain cap")
        first = partition.drop_duplicates("lane", keep="first")
        if len(first) != len(active):
            raise RuntimeError("ordered navigator omitted an active lane")
        stats["iterations"] += 1
        if progress is not None and iteration % 25 == 0:
            progress(dict(iteration=iteration, active=len(active), slices=len(slices),
                          min_path_mm=float(distance[active].min()), max_path_mm=float(distance[active].max()),
                          boundary_relabel_max_mm=float(boundary_relabel.max())))
        names, kinds = first.material.to_numpy(), first.kind.to_numpy()
        extent = first.s1_mm.to_numpy() - first.s0_mm.to_numpy()
        for name in np.unique(names):
            local = np.flatnonzero(names == name)
            lane = active[local]
            before, start, t0 = ke[lane].copy(), distance[lane].copy(), clock[lane].copy()
            length = extent[local].copy()
            material = None if name == "vacuum" else STRAW_GAS if name == STRAW_GAS.name else resolve_material(name)
            if material is not None:
                dedx = np.asarray(eloss.dedx_restricted(material, ledger.momentum(before), 1e9))
                if not np.isfinite(dedx).all() or (dedx < 0).any():
                    raise ValueError("invalid ordered stopping power")
                adaptive = np.divide(fractional_loss * before, dedx,
                                     out=np.full(len(lane), max_step_mm), where=dedx > 0)
                length = np.minimum(length, np.minimum(max_step_mm, adaptive))
                evaluation_ke = before
                if stopping_evaluation == "midpoint":
                    predictor = np.minimum(dedx * length, before - stop_ke_MeV)
                    evaluation_ke = before - .5 * predictor
                    dedx = np.asarray(eloss.dedx_restricted(material, ledger.momentum(evaluation_ke), 1e9))
                    if not np.isfinite(dedx).all() or (dedx < 0).any():
                        raise ValueError("invalid midpoint stopping power")
                to_cutoff = np.divide(before - stop_ke_MeV, dedx,
                                      out=np.full(len(lane), np.inf), where=dedx > 0)
                ends_at_cutoff = length >= to_cutoff
                length = np.minimum(length, to_cutoff)
                debit = np.minimum(dedx * length, before)
                after = np.where(ends_at_cutoff, stop_ke_MeV, before - debit)
                if ((debit > 0) & (after == before)).any():
                    raise RuntimeError("positive material debit cannot advance KE")
            else:
                after = before.copy()
            gas = name == STRAW_GAS.name
            sampled = None
            if sampled_loss and material is not None:
                p_sample = ledger.momentum(evaluation_ke)
                raw = np.asarray(eloss.sample_loss(rng, material, p_sample, length,
                    STRAW_GAS.cuts_e_MeV["tracker"] if gas else 1e9))
                if raw.shape != before.shape or not np.isfinite(raw).all() or (raw < 0).any():
                    raise ValueError("invalid ordered sampled debit")
                sampled = raw.copy()
                if gas:
                    restricted_mean = eloss.dedx_restricted(material, p_sample, STRAW_GAS.cuts_e_MeV["tracker"])
                    raw = raw + np.maximum(dedx - restricted_mean, 0.) * length
                allowed = before - stop_ke_MeV
                fraction = np.divide(allowed, raw, out=np.ones_like(raw), where=raw > allowed)
                fraction = np.minimum(fraction, 1.)
                length = length * fraction
                if gas:
                    sampled = sampled * fraction
                after = np.where(raw >= allowed, stop_ke_MeV, before - raw)
            nav_end = nav_s[lane] + length
            full = length == extent[local]
            nav_end[full] = first.s1_mm.to_numpy()[local[full]]
            end = start + length
            if (length <= 0).any() or (end <= start).any():
                raise RuntimeError("ordered navigation cannot advance")
            dt = linear_kinetic_dt_ns(length, before, after)
            endpos = anchor[lane] + d[lane] * nav_end[:, None]
            terminal = after <= stop_ke_MeV
            eion = np.zeros(len(lane))
            if gas:
                if sampled is None:
                    sampled = np.asarray(eloss.sample_loss(rng, STRAW_GAS, ledger.momentum(evaluation_ke), length,
                                                           STRAW_GAS.cuts_e_MeV["tracker"]))
                if sampled.shape != before.shape or not np.isfinite(sampled).all() or (sampled < 0).any():
                    raise ValueError("invalid ordered gas response sample")
                eion = np.minimum(sampled, before - after)
                stats["gas_capped_slices"] += int(np.count_nonzero(sampled > before - after))
                stats["gas_sampled_MeV"] += float(sampled.sum())
                stats["gas_applied_MeV"] += float(eion.sum())
                js = crossings.point_straw(o[lane] + d[lane] * (.5 * length)[:, None], geom)
                if (js < 0).any():
                    raise RuntimeError("ordered gas interval has no straw owner")
                rows.append(dict(it=lane.copy(), js=js, pos0=o[lane].copy(), pos1=endpos.copy(),
                    dir=d[lane].copy(), path=start, lost=initial[lane] - before, dt=t0,
                    eion=eion, gas_debit=before - after, gas_len=length.copy(),
                    **{k: comp[k][lane].copy() for k in COMPONENTS}))
            slices.append(dict(it=lane.copy(), start=start, end=end, pos0=o[lane].copy(), pos1=endpos.copy(),
                dir=d[lane].copy(), ke0=before, ke1=after, t0=t0, t1=t0 + dt,
                material=np.full(len(lane), name), kind=kinds[local].copy(), stopped=terminal,
                deposited=np.where(gas, eion, np.where(material is None, 0., np.nan)),
                unassigned=np.where(gas, before - after - eion, np.where(material is None, 0., np.nan)),
                escaping=np.full(len(lane), 0. if material is None else np.nan)))
            support = kinds[local] == "support"
            comp["lost_support"][lane] += np.where(support, before - after, 0.)
            comp["lost_thin"][lane] += np.where(~support, before - after, 0.)
            comp["l_support"][lane] += np.where(support, length, 0.)
            comp["l_thin"][lane] += np.where(kinds[local] == "thin", length, 0.)
            ke[lane], distance[lane], clock[lane], o[lane] = after, end, t0 + dt, endpos
            nav_s[lane] = nav_end
            stopped[lane] = terminal
            scatter = np.flatnonzero(~terminal) if material is not None else np.empty(0, np.int64)
            if len(scatter):
                use = lane[scatter]
                x = length[scatter] / eloss.radiation_length_mm(material)
                psc = ledger.momentum(.5 * (before[scatter] + after[scatter]))
                variance = tr.highland_var_increment(psc, xcum[use], xcum[use] + x)
                theta, _, width = _kink_core(rng, variance, np.zeros(len(use)), np.zeros(len(use)))
                if tail:
                    extra = _group_tail(rng, [(material, length[scatter])], np.arange(len(use)),
                                        psc, psc, width, tail_k)
                    stats["tail_scatters"] += int(np.count_nonzero(np.any(extra != 0, axis=1)))
                    theta += extra
                _, d[use] = _kink(o[use], d[use], theta, np.zeros_like(theta))
                changed = use[np.any(theta != 0, axis=1)]
                anchor[changed], nav_s[changed] = o[changed], 0.
                dirty[changed] = True
                xcum[use] += x
                stats["kinks"] += len(changed)
    else:
        stats["capped"] = int(np.count_nonzero(~stopped & (distance < max_path_mm)))
        raise RuntimeError("ordered interior iteration cap; no transmission inferred")
    stats["stopped"] = int(stopped.sum())
    interior.last_lookahead_evidence = pd.concat(lookahead_records, ignore_index=True) if lookahead_records else pd.DataFrame()
    stats["bounded_forward_lookahead"] = bounded_forward_lookahead
    stats["bounded_lookahead_queries"] = len(interior.last_lookahead_evidence)
    interior.last_endpoint_evidence = pd.concat(endpoint_records, ignore_index=True) if endpoint_records else pd.DataFrame()
    interior.last_leading_source_evidence = pd.concat(leading_records, ignore_index=True) if leading_records else pd.DataFrame()
    stats["leading_signed_sources"] = len(interior.last_leading_source_evidence)
    stats["signed_endpoint_budget_mm"] = signed_endpoint_budget_mm
    stats["signed_endpoints_shifted"] = len(interior.last_endpoint_evidence)
    stats["signed_max_shift_mm"] = (float(interior.last_endpoint_evidence.shift_mm.abs().max())
                                     if len(interior.last_endpoint_evidence) else 0.)
    interior.last_stats = stats
    interior.last_final = dict(o=o, d=d, p=ledger.momentum(ke), path=distance, xcum=xcum,
                               ke=ke, stopped=stopped, dt=clock, boundary_relabel_mm=boundary_relabel, **comp)
    interior.last_slices = ({k: np.concatenate([s[k] for s in slices]) for k in slices[0]} if slices else {})
    interior.last_kinks = dict(it=np.empty(0, np.int64), end=np.empty((0, 3)), pos=np.empty((0, 3)),
                               dir=np.empty((0, 3)), path=np.empty(0), p=np.empty(0))
    if not rows:
        return dict(it=np.empty(0, np.int64), js=np.empty(0, np.int64), pos0=np.empty((0, 3)),
                    pos1=np.empty((0, 3)), dir=np.empty((0, 3)), path=np.empty(0), lost=np.empty(0),
                    dt=np.empty(0), eion=np.empty(0), gas_debit=np.empty(0), gas_len=np.empty(0),
                    **{k: np.empty(0) for k in COMPONENTS})
    out = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    order = np.lexsort((out["path"], out["it"]))
    out = {k: v[order] for k, v in out.items()}
    frame = pd.DataFrame({"it": out["it"], "js": out["js"], "row": np.arange(len(order)),
        "gas_len": out["gas_len"],
        "eion": out["eion"], "gas_debit": out["gas_debit"]})
    grouped = frame.groupby(["it", "js"], sort=False)
    first, last = grouped.row.first().to_numpy(), grouped.row.last().to_numpy()
    merged = {k: v[first] for k, v in out.items()}
    merged["pos1"] = out["pos1"][last]
    for name in ("gas_len", "eion", "gas_debit"):
        merged[name] = grouped[name].sum().to_numpy()
    return merged


def vertices(mu, selection, slices, final, t_start):
    """Actual traversed endpoints only, with the slice KE law for surface queries."""
    from .steps import C_MM_PER_NS, KEY
    if not slices:
        columns = ["run", "subrun", "event", "sp", "path", "p", "x", "y", "z", "ux", "uy", "uz",
                   "kinetic_MeV", "time_ns", "beta", "t0"]
        return pd.DataFrame({k: np.empty(0) for k in columns})
    lane = np.r_[slices["it"], np.arange(len(selection))]
    keys = selection[lane]
    out = pd.DataFrame({k: mu[k].to_numpy()[keys] for k in KEY})
    out["sp"] = mu.mukey.to_numpy()[keys]
    out["path"] = np.r_[slices["start"], final["path"]]
    out["p"] = ledger.momentum(np.r_[slices["ke0"], final["ke"]])
    pos = geometry.to_detector(np.r_[slices["pos0"], final["o"]])
    dirs = np.r_[slices["dir"], final["d"]]
    for j, axis in enumerate("xyz"):
        out[axis], out[f"u{axis}"] = pos[:, j], dirs[:, j]
    out["kinetic_MeV"] = np.r_[slices["ke0"], final["ke"]]
    out["time_ns"] = t_start[keys] + np.r_[slices["t0"], final["dt"]]
    out["beta"] = ledger.beta(out.p.to_numpy())
    out["t0"] = out.time_ns - out.path / (np.maximum(out.beta, 1e-12) * C_MM_PER_NS)
    return out.sort_values(KEY + ["sp", "path"], kind="stable").reset_index(drop=True)
