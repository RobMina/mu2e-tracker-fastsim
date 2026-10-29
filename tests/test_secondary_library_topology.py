"""Exact-manifest and fail-closed library annotation regression tests."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from test_anchor_topology import _source
from test_library_v2 import _build
from trkfastsim import library as L

_spec = importlib.util.spec_from_file_location('secondary_library_topology',
    Path(__file__).resolve().parents[1] / 'scripts/secondary_library_topology.py')
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


@pytest.mark.parametrize('metadata', [{'inputs': '../bad.root', 'nfiles': 1},
    {'inputs': 'a.root;a.root', 'nfiles': 2}, {'inputs': 'a.root', 'nfiles': 2}])
def test_manifest_fail_closed(metadata):
    with pytest.raises(ValueError):
        M.donor_paths(metadata, '/source')


def test_manifest_preserves_order():
    assert M.donor_paths({'inputs': 'b.root;a.root', 'nfiles': 2}, '/source') == [
        Path('/source/b.root'), Path('/source/a.root')]


def test_chunked_builder_matches_full_payload(monkeypatch):
    g, s, p = _source()
    s2, p2 = s.assign(subrun=2), p.assign(subrun=2)
    full = _build(pd.concat([s, s2]), pd.concat([p, p2]), g, topology=True)
    original = L.build
    def one(paths, geom, topology=False):
        assert topology
        from trkfastsim import tables as T
        a, b = (s, p) if paths == ['a.root'] else (s2, p2)
        with monkeypatch.context() as ctx:
            ctx.setattr(T, 'read_many', lambda paths, tree, cols=None: (a if tree == 'sgsteps' else b)[cols or slice(None)].copy())
            return original(paths, geom, topology=topology)
    monkeypatch.setattr(L, 'build', one)
    bounded = M.build_bounded([Path('a.root'), Path('b.root')], g)
    assert M.require_legacy_identity(full, bounded)['events'] == 2
    bounded['offaxis'].loc[0, 'eion'] += 1.
    with pytest.raises(ValueError, match='legacy payload'):
        M.require_legacy_identity(full, bounded)


def test_annotation_preserves_original_derived_roundoff_but_rejects_payload_changes():
    import numpy as np
    g, s, p = _source()
    base, annotated = _build(s, p, g), _build(s, p, g, topology=True)
    annotated['instraw'].loc[0, 'p'] = np.nextafter(base['instraw'].loc[0, 'p'], np.inf)
    result = M.retain_payload(base, annotated)
    M.require_legacy_identity(base, result)
    assert result['instraw'].loc[0, 'p'] == base['instraw'].loc[0, 'p']
    annotated['offaxis'].loc[0, 'eion'] = np.nextafter(base['offaxis'].loc[0, 'eion'], np.inf)
    with pytest.raises(ValueError, match='annotation payload'):
        M.retain_payload(base, annotated)
    annotated['offaxis'] = _build(s, p, g, topology=True)['offaxis']
    annotated['events'].loc[0, 'p'] += 1
    with pytest.raises(ValueError, match='roundoff allowance'):
        M.retain_payload(base, annotated)
