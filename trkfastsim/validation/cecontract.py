"""CE denominator identity and efficiency-claim safeguards; no transport or physics changes.

A complete-birth claim describes the externally counted input population only. It is not an equivalence verdict,
a production/POT normalization, or certification of calorimeter, secondary or return-passage acceptance.
"""
import hashlib

import numpy as np
import pandas as pd

from mu2efastsim import tables

KEY = tables.KEY


def integer_keys(d, label):
    """Normalize integer storage without float promotion in mixed signed/unsigned pandas joins."""
    if (d[KEY].isna().any().any() or (d[KEY] < 0).any().any()
            or not all(pd.api.types.is_integer_dtype(d[k].dtype) for k in KEY)):
        raise ValueError(f"invalid event identities: {label}")
    d = d.copy()
    d[KEY] = d[KEY].astype(np.uint64)
    return d


def unique(d, label):
    """Reject duplicate or missing identities before any join."""
    d = integer_keys(d, label)
    if d.duplicated(KEY).any():
        raise ValueError(f"duplicate event identities: {label}")
    return d.sort_values(KEY).reset_index(drop=True)


def subset(d, denominator, label):
    """Require unique exact event keys within the original birth denominator; missing rows remain allowed."""
    d = unique(d, label)
    den = unique(denominator[KEY], "birth denominator")
    membership = d[KEY].merge(den, on=KEY, how="left", indicator=True, validate="one_to_one")
    if (membership._merge != "both").any():
        raise ValueError(f"event outside birth denominator: {label}")
    return d


def fingerprint(d):
    """Hash sorted uint64 run/subrun/event rows in fixed little-endian order, independent of storage dtype."""
    keys = unique(d[KEY], "birth fingerprint")[KEY].to_numpy(dtype="<u8")
    return hashlib.sha256(keys.tobytes(order="C")).hexdigest()


def selection_contract(events, selection, expected_starts=None):
    """An unfiltered label requires a positive externally audited count, not just a selected table length."""
    if selection not in ("selected-dts", "unfiltered"):
        raise ValueError("unknown reference selection")
    if selection == "unfiltered" and (
            isinstance(expected_starts, (bool, np.bool_))
            or not isinstance(expected_starts, (int, np.integer))
            or expected_starts <= 0 or events != expected_starts):
        raise ValueError("unfiltered reference requires its externally audited complete event count")
    return dict(reference_selection=selection, conditional_on_dts_selection=selection != "unfiltered",
                expected_starts=expected_starts)


def efficiency_scope(denominator, provenance, claim="diagnostic"):
    """Fail closed on incomplete/contradictory receipts and unsupported complete-birth claims."""
    den = unique(denominator[KEY], "birth denominator")
    if den.empty:
        raise ValueError("empty birth denominator")
    if claim not in ("diagnostic", "complete-birth"):
        raise ValueError("unknown efficiency claim")
    selection = provenance.get("reference_selection", "selected-dts")
    contract = selection_contract(len(den), selection, provenance.get("expected_starts"))
    conditional = provenance.get("conditional_on_dts_selection", contract["conditional_on_dts_selection"])
    if not isinstance(conditional, bool) or conditional != contract["conditional_on_dts_selection"]:
        raise ValueError("contradictory reference selection receipt")
    if "generated_starts" in provenance and provenance["generated_starts"] != len(den):
        raise ValueError("receipt generated-start count differs from birth denominator")
    digest = fingerprint(den)
    if "birth_keys_sha256" in provenance and provenance["birth_keys_sha256"] != digest:
        raise ValueError("receipt birth identities differ from birth denominator")
    mode = provenance.get("mode", "unknown")
    if mode not in ("entry", "chain", "unknown"):
        raise ValueError("unknown transport mode in receipt")
    eligible = (selection == "unfiltered" and mode == "chain"
                and provenance.get("generated_starts") == len(den))
    if claim == "complete-birth" and not eligible:
        raise ValueError("complete-birth efficiency requires an unfiltered chain receipt with audited complete count")
    interpretation = ("shared-G4-entry tracker control" if mode == "entry" else
                      "complete-birth chain estimate" if eligible else "conditional or unverified-input diagnostic")
    return dict(**contract, transport_mode=mode, requested_claim=claim,
                complete_birth_eligible=eligible, interpretation=interpretation,
                denominator_events=len(den), birth_keys_sha256=digest,
                receipt_identity_binding="sha256" if "birth_keys_sha256" in provenance else "legacy count-only or absent",
                production_filter_parity="not certified",
                validity_verdict="not evaluated; scope eligibility is not physics closure",
                population="audited input birth population only; not production-wide or POT-normalized")
