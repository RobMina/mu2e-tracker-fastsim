"""Read-only admission of a finished CE confirmation reference from three audit reports.

Admission checks report evidence, not the underlying art products or physics. It is neither physics closure,
lossless-retention certification, production-filter parity nor production/POT normalization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


SHA256 = re.compile(r"[0-9a-fA-F]{64}")
CYCLE = re.compile(r"Events;([1-9][0-9]*)")


def _object(value, label):
    if type(value) is not dict:
        raise ValueError(f"{label}: JSON object required")
    return value


def _field(report, key, label):
    if key not in report:
        raise ValueError(f"{label}.{key}: required field missing")
    return report[key]


def _count(report, key, label, maximum=None):
    value = _field(report, key, label)
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{label}.{key}: nonnegative integer count within bounds required")
    return value


def _true(report, key, label):
    if _field(report, key, label) is not True:
        raise ValueError(f"{label}.{key}: true boolean required")


def _digest(report, key, label):
    value = _field(report, key, label)
    if type(value) is not str or SHA256.fullmatch(value) is None:
        raise ValueError(f"{label}.{key}: 64-hex SHA256 required")
    return value


def admit(census, source, primary):
    """Fail closed on incomplete, mistyped or contradictory audit evidence; perform no file writes."""
    for label, report in (("census", census), ("source", source), ("primary", primary)):
        _object(report, label)
    n = _count(census, "events", "census")
    if n == 0:
        raise ValueError("census.events: positive count required")
    for report, label, keys in (
            (source, "source", ("events",)),
            (primary, "primary", ("generated_starts", "external_expected_events", "primary_electrons")),
            (census, "census", ("status_zero_events", "inclusive_path_pass_events"))):
        for key in keys:
            if _count(report, key, label) != n:
                raise ValueError(f"{label}.{key}: must equal census.events")
    for report, label, keys in (
            (census, "census", ("generation_saved_metadata_exact", "physics_eligible")),
            (source, "source", ("exact_sequential_order", "source_window_complete")),
            (primary, "primary", ("closed_census_keys_exact",))):
        for key in keys:
            _true(report, key, label)
    for report, label, keys in (
            (census, "census", ("validity_failure_events",)),
            (primary, "primary", ("primary_step_events_without_vd13",
                                    "steps_before_first_vd13_in_crossing_events"))):
        for key in keys:
            if _count(report, key, label) != 0:
                raise ValueError(f"{label}.{key}: zero required")

    statuses = _object(_field(census, "status_counts", "census"), "census.status_counts")
    if set(statuses) != {"0"} or _count(statuses, "0", "census.status_counts") != n:
        raise ValueError("census.status_counts: all events must have native status zero")
    selected = _count(census, "stock_selected_events", "census", n)
    rejected = _count(census, "stock_rejected_events", "census", n)
    if selected + rejected != n:
        raise ValueError("census: stock selected/rejected counts must partition all events")
    raw = _object(_field(census, "raw_products", "census"), "census.raw_products")
    _true(raw, "all_required_products_present", "census.raw_products")
    cycles = _object(_field(census, "event_cycles", "census"), "census.event_cycles")
    if not cycles or any(type(key) is not str or CYCLE.fullmatch(key) is None for key in cycles):
        raise ValueError("census.event_cycles: nonempty Events;positive-cycle mapping required")
    for key in cycles:
        _count(cycles, key, "census.event_cycles", n)
    highest = max(cycles, key=lambda key: int(CYCLE.fullmatch(key).group(1)))
    if cycles[highest] != n:
        raise ValueError("census.event_cycles: highest cycle must contain all events")

    birth = _digest(census, "birth_keys_sha256", "census")
    if (_digest(primary, "birth_keys_sha256", "primary") != birth
            or _digest(source, "generated_keys_sha256", "source") != birth):
        raise ValueError("census/source/primary: generated birth identity digests differ")
    _digest(source, "source_keys_sha256", "source")
    _digest(primary, "census_events_sha256", "primary")
    for report, label, keys in (
            (census, "census", ("art", "generation_log", "saved_log")),
            (source, "source", ("source_log", "window_art"))):
        hashes = _object(_field(report, "input_sha256", label), f"{label}.input_sha256")
        for key in keys:
            _digest(hashes, key, f"{label}.input_sha256")

    vd13 = _count(primary, "primary_vd13", "primary", n)
    step_events = _count(primary, "primary_step_events", "primary", vd13)
    steps = _count(primary, "primary_steps", "primary")
    all_steps = _count(primary, "all_steps", "primary")
    other = _count(primary, "other_particle_steps", "primary")
    if steps < step_events or (steps == 0) != (step_events == 0) or steps + other != all_steps:
        raise ValueError("primary: inconsistent step/event/other-particle counts")
    return dict(admitted=True, events=n, birth_keys_sha256=birth,
                stock_selected_events=selected, stock_rejected_events=rejected,
                scope="finished confirmation reference admission from audit reports only",
                validity_verdict="not evaluated; admission is not physics closure",
                limitations="no underlying-product re-audit, lossless-retention certification, "
                            "production-filter parity or production/POT normalization")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("census", "source", "primary", "out"):
        ap.add_argument(f"--{name}", required=True)
    a = ap.parse_args(argv)
    reports, hashes = {}, {}
    for label in ("census", "source", "primary"):
        # Hash exactly the bytes parsed, not a later re-read or a reserialized object.
        data = Path(getattr(a, label)).read_bytes()
        hashes[label] = hashlib.sha256(data).hexdigest()
        reports[label] = json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant)
    report = admit(**reports)
    report["input_sha256"] = hashes
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with Path(a.out).open("x") as stream:
        stream.write(serialized)
    print(serialized, end="")
    return report


if __name__ == "__main__":
    main()
