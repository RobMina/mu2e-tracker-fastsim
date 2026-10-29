"""Vectorized captured-transition census, not a replay or constructor-history claim.

Tables are flat Awkward record arrays using f1_native_decision_audit's full
(run, subrun, event, iseed[, iteration_ordinal]) identities. Malformed identities,
shapes, counts and joins raise ValueError. Observed owner inequality and object
absence/reappearance are reported, never silently excluded. Every public result
has a JSON-friendly ``census``; other entries are NumPy arrays for parent gating.
No event, iteration, object or hit loops are used.
"""
import awkward as ak
import numpy as np

from f1_native_decision_audit import KEYS, ITER_KEYS, array, column, dense, keys, match_rows, require

LABELS = ("same_meta", "seed_meta", "extension_entry", "extension_meta")
SOURCE_KEYS = KEYS + ("input_index",)
SNAPSHOT_KEYS = ITER_KEYS + ("input_index",)


def _optional_join(wanted, available):
    """Full int64 key join with -1 for absence; available identities must be unique."""
    require(wanted.ndim == available.ndim == 2 and wanted.shape[1] == available.shape[1], "identity shapes differ")
    require(len(np.unique(available, axis=0)) == len(available), "duplicate available identities")
    if not len(available):
        return np.full(len(wanted), -1, dtype=np.int64)
    dtype = np.dtype([(f"k{i}", np.int64) for i in range(wanted.shape[1])])
    w = np.ascontiguousarray(wanted, dtype=np.int64).view(dtype).ravel()
    a = np.ascontiguousarray(available, dtype=np.int64).view(dtype).ravel()
    order = np.argsort(a)
    loc = np.searchsorted(a[order], w)
    safe = np.minimum(loc, len(a)-1)
    return np.where((loc < len(a)) & (a[order[safe]] == w), order[safe], -1)


def _integer(table, name, minimum=0):
    out = column(table, name, "iu")
    require((out >= minimum) & (out <= np.iinfo(np.int64).max), name + ": invalid integer")
    return out.astype(np.int64)


def _identity(table, iterations, ik):
    parent = match_rows(keys(table, ITER_KEYS, unique=False), ik)
    for field in ("miter", "iter"):
        require(_integer(table, field) == _integer(iterations, field)[parent], field + ": parent identity mismatch")
    return parent


def iteration_links(iterations, seed_stages=4, extension_stages=9, miter_origin=0):
    """Return ALL captured adjacent-ordinal links, including unswapped parents.

    ``previous``, ``following`` are original table row indices, ``labels`` is a
    string per link and ``swapped`` is the previous-row bool (a consumer gate,
    NOT a selection). ``terminal`` contains rows with no next captured ordinal.
    Ordinals must start at zero and be contiguous within each full target.
    miter is in [miter_origin, miter_origin+seed_stages+extension_stages); both stage arguments are counts.
    Only the four LABELS transitions are allowed. history_index, if present,
    is checked as a nonnegative integer but its native semantics belong to the
    parent audit; ordinal adjacency does not claim complete native history.
    """
    require(type(seed_stages) is int and seed_stages > 0 and type(extension_stages) is int
            and extension_stages >= 0 and type(miter_origin) is int and miter_origin >= 0, "invalid stage counts")
    ik = keys(iterations, ITER_KEYS)
    require(len(ik) > 0, "empty iteration capture")
    require(ik[:, 4] < np.iinfo(np.int64).max, "ordinal cannot be incremented")
    _, target, counts = np.unique(ik[:, :4], axis=0, return_inverse=True, return_counts=True)
    order = np.lexsort((ik[:, 4], target))
    require(ik[order, 4] == np.arange(len(ik))-np.repeat(np.cumsum(counts)-counts, counts), "iteration ordinal gaps")
    raw_miter, iteration = (_integer(iterations, f) for f in ("miter", "iter"))
    require(raw_miter >= miter_origin, "miter below configured origin")
    miter = raw_miter-miter_origin
    require(miter < seed_stages+extension_stages, "miter outside configured stages")
    first = ik[:, 4] == 0
    require((miter[first] == 0) & (iteration[first] == 0), "initial iteration outside first configured stage")
    require(iteration < np.iinfo(np.int64).max, "iter cannot be incremented")
    swapped = column(iterations, "swapped", "b")
    if "history_index" in iterations.fields:
        _integer(iterations, "history_index")
    wanted = ik.copy()
    wanted[:, 4] += 1
    nextrow = _optional_join(wanted, ik)
    previous = np.flatnonzero(nextrow >= 0)
    following = nextrow[previous]
    p, n = miter[previous], miter[following]
    reset = iteration[following] == 0
    categories = np.column_stack((
        (n == p) & (iteration[following] == iteration[previous]+1),
        (n == p+1) & (n < seed_stages) & reset,
        (p == seed_stages-1) & (n == seed_stages) & reset,
        (p >= seed_stages) & (n == p+1) & reset))
    require(categories.sum(axis=1) == 1, "unknown adjacent transition")
    labels = np.asarray(LABELS)[np.argmax(categories, axis=1)]
    terminal = np.flatnonzero(nextrow < 0)
    census = dict(n_iterations=len(ik), n_targets=len(counts), n_links=len(previous),
                  n_terminal=len(terminal), terminal_key_fields=list(ITER_KEYS), terminal_keys=ik[terminal].tolist(),
                  n_swapped=int(swapped[previous].sum()), n_unswapped=int((~swapped[previous]).sum()),
                  by_label={label: int((labels == label).sum()) for label in LABELS},
                  seed_stages=seed_stages, extension_stages=extension_stages, miter_origin=miter_origin,
                  all_existing_next_rows_selected=True, complete_native_history_claim=False,
                  history_index_semantics_checked=False)
    return dict(previous=previous, following=following, labels=labels, swapped=swapped[previous],
                terminal=terminal, census=census)


def _link_rows(links, ik):
    p, n = (array(links[k]) for k in ("previous", "following"))
    require(p.ndim == n.ndim == 1 and p.shape == n.shape and p.dtype.kind in "iu"
            and n.dtype.kind in "iu", "link index shape/dtype")
    require((p >= 0) & (p < len(ik)) & (n >= 0) & (n < len(ik)), "link index bounds")
    require(len(np.unique(p)) == len(p) and len(np.unique(n)) == len(n), "duplicate links")
    require(np.all(ik[p, :4] == ik[n, :4], axis=1) & (ik[n, 4] == ik[p, 4]+1), "link full identity mismatch")
    return p.astype(np.int64), n.astype(np.int64)


def owner_continuity(pieces, iterations, links):
    """Compare previous owner=1 with next owner=0 by full key AND piece_index.

    ``equality``, ``counts_previous``, ``counts_following``, ``count_equal``,
    ``missing_candidate`` and each ``field_equal`` entry have shape (n_links,).
    Missing candidates (even zero/zero on unswapped links) are NOT matches.
    Fields parameters[6], covariance[36], range[2], bnom[3], mass, charge must
    have exactly equal finite components; nonfinite values report inequality.
    Unequal owner counts are observations, but disagreement with declared
    nold/ncandidate, duplicate indices or holes is malformed and raises.
    census records both full iteration keys for every unequal link. All owners,
    including terminal and unswapped rows, receive identity/count validation.
    """
    ik = keys(iterations, ITER_KEYS)
    p, n = _link_rows(links, ik)
    pk = keys(pieces, ITER_KEYS + ("owner", "piece_index"))
    parent = _identity(pieces, iterations, ik)
    owner, index = pk[:, 5], pk[:, 6]
    require(np.isin(owner, (0, 1)), "unknown piece owner")
    counts = np.column_stack([_integer(iterations, f) for f in ("nold", "ncandidate")])
    if "candidate_present" in iterations.fields:
        require(column(iterations, "candidate_present", "b") == (counts[:, 1] > 0), "candidate presence mismatch")
    require(index < counts[parent, owner], "piece_index outside owner count")
    for which in (0, 1):
        require(np.bincount(parent[owner == which], minlength=len(ik)) == counts[:, which], "piece count mismatch/holes")
    before, after = counts[p, 1], counts[n, 0]
    count_equal, missing = before == after, before == 0
    comparable = count_equal & ~missing
    link_of_parent = np.full(len(ik), -1, dtype=np.int64)
    link_of_parent[p] = np.arange(len(p))
    rows = np.flatnonzero((owner == 1) & (link_of_parent[parent] >= 0))
    lane = link_of_parent[parent[rows]]
    wanted = np.column_stack((ik[n[lane]], np.zeros(len(rows), dtype=np.int64), index[rows]))
    following = _optional_join(wanted, pk)
    joined = following >= 0
    field_equal, nonfinite = {}, {}
    for field, width in (("parameters", 6), ("covariance", 36), ("range", 2), ("bnom", 3), ("mass", 1), ("charge", 1)):
        values = dense(pieces, field, width, finite=False) if width > 1 else column(pieces, field, "fiu")[:, None]
        finite = np.all(np.isfinite(values), axis=1)
        equal_piece = np.zeros(len(rows), dtype=bool)
        equal_piece[joined] = (finite[rows[joined]] & finite[following[joined]] &
                               np.all(values[rows[joined]] == values[following[joined]], axis=1))
        equal = comparable.copy()
        np.logical_and.at(equal, lane, equal_piece)
        field_equal[field] = equal
        nonfinite[field] = pk[~finite].tolist()
    equality = np.logical_and.reduce(list(field_equal.values()))
    bad = ~equality
    census = dict(n_links=len(p), n_equal=int(equality.sum()), n_unequal=int(bad.sum()),
                  n_count_mismatch=int((~count_equal).sum()), n_missing_candidate=int(missing.sum()),
                  n_missing_candidate_unswapped=int((missing & ~column(iterations, "swapped", "b")[p]).sum()),
                  iteration_key_fields=list(ITER_KEYS), unequal_previous_keys=ik[p[bad]].tolist(),
                  unequal_following_keys=ik[n[bad]].tolist(),
                  count_mismatch_previous_keys=ik[p[~count_equal]].tolist(),
                  count_mismatch_following_keys=ik[n[~count_equal]].tolist(),
                  missing_candidate_previous_keys=ik[p[missing]].tolist(),
                  field_unequal_previous_keys={f: ik[p[~v]].tolist() for f, v in field_equal.items()},
                  field_unequal_following_keys={f: ik[n[~v]].tolist() for f, v in field_equal.items()},
                  piece_key_fields=list(ITER_KEYS + ("owner", "piece_index")), nonfinite_piece_keys=nonfinite,
                  exact_finite_comparison=True, replay_gate_applied=False)
    return dict(equality=equality, counts_previous=before, counts_following=after, count_equal=count_equal,
                missing_candidate=missing, field_equal=field_equal, census=census)


def _seed_family(seeds, sk, prefix):
    fields = ("index", "shindex", "strawid") if prefix == "sh" else ("index", "strawid", "partner_shindex")
    values, counts = {}, None
    for field in fields:
        name = prefix + "_" + field
        require(seeds[name].ndim == 2, name + ": ragged shape")
        size = array(ak.num(seeds[name], axis=1))
        if counts is None:
            counts = size
        require(np.array_equal(size, counts), name + ": ragged lengths differ")
        v = array(ak.flatten(seeds[name], axis=1))
        require(v.ndim == 1 and (v.dtype.kind in "iu" or len(v) == 0), name + ": integer shape/dtype")
        require((v >= (-1 if field == "partner_shindex" else 0)) & (v <= np.iinfo(np.int64).max), name + ": invalid identity")
        values[field] = v.astype(np.int64)
    source = np.column_stack((np.repeat(sk, counts, axis=0), values["index"]))
    require(len(np.unique(source, axis=0)) == len(source), "duplicate seed source identities")
    return source, values


def object_lifetimes(effects, iterations, seeds, seed_stages=4, miter_origin=0):
    """Census straw hits (identity_kind=1) and crossings (5), never other families.

    ``hits`` and ``crossings`` contain source_keys (KEYS+input_index), first_rows
    (original effects rows), first_ordinal, first_miter, last_ordinal, n_snapshots,
    initial and extension masks, shindex/strawid/partner_shindex where relevant.
    Source families are joined separately: constraint input_index collisions are
    legal. Source shindex/strawid and final crossing partner must stay constant.
    Initial membership is an exact seed sh_index/sx_index join within captured targets, not an index cut.
    New objects must first appear at miter=miter_origin+seed_stages. Disappearance over a
    captured link and subsequent reappearance are reported, NOT rejected.

    partner_shindex is FINAL-COLLECTION metadata, NOT an iteration-local link.
    All effects must declare identity_from_final_collections=True. Each associated
    crossing must join a unique captured hit's final source shindex, same target
    and straw. Such a hit need not exist in that crossing's first snapshot.
    Initial crossing metadata must also agree with sx_partner_shindex. No
    extension constructor offsets, constructor timing or partner timing is inferred.
    The JSON-friendly census records full keys; returned NumPy masks are aligned
    to each family's unique source_keys, not to the effects table.
    """
    links = iteration_links(iterations, seed_stages=seed_stages, miter_origin=miter_origin)
    extension_start = seed_stages+miter_origin
    ik, sk = keys(iterations, ITER_KEYS), keys(seeds)
    all_seed_count = len(sk)
    captured_targets = np.unique(ik[:, :4], axis=0)
    selected_seeds = match_rows(captured_targets, sk)
    seeds, sk = seeds[selected_seeds], sk[selected_seeds]
    parent = _identity(effects, iterations, ik)
    require(column(effects, "identity_from_final_collections", "b"), "identity must come from final collections")
    kind = _integer(effects, "identity_kind")
    has_next = np.zeros(len(ik), dtype=bool)
    has_next[links["previous"]] = True
    result, summaries = {}, {}
    for name, code, prefix in (("hits", 1, "sh"), ("crossings", 5, "sx")):
        original = np.flatnonzero(kind == code)
        table = effects[original]
        snap = keys(table, SNAPSHOT_KEYS)
        source = np.column_stack((snap[:, :4], snap[:, 5]))
        unique, inverse, counts = np.unique(source, axis=0, return_inverse=True, return_counts=True)
        order = np.lexsort((snap[:, 4], inverse))
        start = np.r_[0, np.cumsum(counts)[:-1]] if len(counts) else np.empty(0, dtype=np.int64)
        first, last = order[start], order[np.cumsum(counts)-1]
        values = {"strawid": _integer(table, "strawid")}
        values["shindex" if code == 1 else "partner_shindex"] = _integer(table, "shindex" if code == 1 else "partner_shindex", 0 if code == 1 else -1)
        for field, value in values.items():
            require(value == value[first][inverse], "source identity changed across snapshots: " + field)
        seed_source, seed_values = _seed_family(seeds, sk, prefix)
        sr = _optional_join(unique, seed_source)
        initial = sr >= 0
        for field in values:
            require(values[field][first[initial]] == seed_values[field][sr[initial]], "initial seed identity mismatch: " + field)
        first_miter = _integer(table, "miter")[first]
        require(first_miter[~initial] == extension_start, "new object first seen outside extension entry")
        wanted = snap.copy()
        wanted[:, 4] += 1
        nextrow = _optional_join(wanted, snap)
        disappear = has_next[parent[original]] & (nextrow < 0)
        reappear = np.zeros(len(table), dtype=bool)
        reappear[order[1:]] = ((inverse[order[1:]] == inverse[order[:-1]]) &
                              (snap[order[1:], 4] > snap[order[:-1], 4]+1))
        obj = dict(source_keys=unique, first_rows=original[first], first_ordinal=snap[first, 4],
                   first_miter=first_miter, last_ordinal=snap[last, 4], n_snapshots=counts,
                   initial=initial, extension=~initial,
                   disappearance_keys=snap[disappear], reappearance_keys=snap[reappear])
        obj.update({field: v[first] for field, v in values.items()})
        result[name] = obj
        summaries[name] = dict(n_snapshots=len(table), n_objects=len(unique), n_initial=int(initial.sum()),
                               n_extension=int((~initial).sum()), n_first_extension=int((first_miter == extension_start).sum()),
                               n_first_extension_new=int(((~initial) & (first_miter == extension_start)).sum()),
                               extension_source_keys=unique[~initial].tolist(),
                               source_key_fields=list(SOURCE_KEYS), source_keys=unique.tolist(),
                               first_ordinal=obj["first_ordinal"].tolist(), first_miter=first_miter.tolist(),
                               last_ordinal=obj["last_ordinal"].tolist(), snapshots_per_object=counts.tolist(),
                               identities={field: v[first].tolist() for field, v in values.items()},
                               initial=initial.tolist(), n_initial_first_seen_after_zero=int((initial & (obj["first_ordinal"] != 0)).sum()),
                               n_seed_objects_not_observed=int((_optional_join(seed_source, unique) < 0).sum()),
                               snapshot_key_fields=list(SNAPSHOT_KEYS), n_disappearances=int(disappear.sum()),
                               disappearance_keys=snap[disappear].tolist(), n_reappearances=int(reappear.sum()),
                               reappearance_keys=snap[reappear].tolist(), no_disappearance_or_reappearance=not bool(disappear.any() or reappear.any()))
    hits, crossings = result["hits"], result["crossings"]
    hit_identity = np.column_stack((hits["source_keys"][:, :4], hits["shindex"]))
    require(len(np.unique(hit_identity, axis=0)) == len(hit_identity), "duplicate final hit shindex identity")
    associated = crossings["partner_shindex"] >= 0
    wanted = np.column_stack((crossings["source_keys"][associated, :4], crossings["partner_shindex"][associated]))
    partner = match_rows(wanted, hit_identity)
    require(hits["strawid"][partner] == crossings["strawid"][associated], "crossing partner straw differs")
    partner_rows = np.full(len(associated), -1, dtype=np.int64)
    partner_rows[associated] = partner
    crossings["partner_hit_rows"] = partner_rows  # indexes unique hits, NOT snapshot rows
    partner_seen_later = np.zeros(len(associated), dtype=bool)
    partner_seen_later[associated] = hits["first_ordinal"][partner] > crossings["first_ordinal"][associated]
    summaries["crossings"].update(n_associated=int(associated.sum()), n_unassociated=int((~associated).sum()),
                                  n_extension_associated=int((crossings["extension"] & associated).sum()),
                                  n_extension_unassociated=int((crossings["extension"] & ~associated).sum()),
                                  n_final_partner_first_seen_later=int(partner_seen_later.sum()),
                                  final_partner_first_seen_later_source_keys=crossings["source_keys"][partner_seen_later].tolist())
    result["census"] = dict(**summaries, n_all_seed_targets=all_seed_count, n_captured_seed_targets=len(sk),
                             n_seed_targets_outside_iteration_capture=all_seed_count-len(sk),
                             identity_basis="final_collections", crossing_partner_basis="final_collections_not_snapshot",
                             final_partner_join_population="unique captured hit source identities stamped from final collections",
                             snapshot_partner_association_claim=False, initial_constructor_offsets_compared=False,
                             extension_constructor_offsets_captured=False, constructor_history_claim=False,
                             limits=["Final crossing partners are not iteration-local associations.",
                                     "Initial seed offsets may be captured but are not examined here.",
                                     "Extension constructor offsets/history are absent; no constructor replay claim.",
                                     "Captured lifetime continuity does not certify unrecorded native history."])
    return result
