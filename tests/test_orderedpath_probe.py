"""Probe report preserves navigation refusal without calling it production admission."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import orderedpath_probe as P


def test_probe_records_refusal_and_tuple_keyed_omissions(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    for name in ('capture.npz', 'complete.json', 'geom.gdml'):
        (source / name).write_bytes(b'synthetic')
    state = dict(identity=np.array([[1, 2, 3, 4]]), position_mm=np.zeros((1, 3)),
                 direction=np.array([[1., 0, 0]]), model_ke_MeV=np.array([20.]))
    monkeypatch.setattr(P, 'load_capture', lambda p: ({'arm': 'AMT'}, {'interior': (None, {})}))
    monkeypatch.setattr(P, 'flatten', lambda *a: state)
    monkeypatch.setattr(P.geometry, 'from_gdml', lambda p: None)
    monkeypatch.setattr(P.steps, 'TrackerInterior', lambda *a: SimpleNamespace(
        shells=None, calo_skipped={('torus', 'example'): 2}))
    def refused(*a, **kw):
        raise ValueError('strict signed overlap')
    monkeypatch.setattr(P.orderedpath, 'transport', refused)
    out = tmp_path / 'result'
    result = P.run(source, source / 'geom.gdml', out, '1' * 40, '2' * 40)
    assert result['status'] == 'refused'
    assert result['reason'] == 'strict signed overlap'
    assert result['admitted_for_production'] is False
    assert result['skipped_calorimeter_solids'][0]['count'] == 2
    assert json.loads((out / 'complete.json').read_text())['identities'] == [[1, 2, 3, 4]]
    with pytest.raises(FileExistsError):
        P.run(source, source / 'geom.gdml', out, '1' * 40, '2' * 40)
