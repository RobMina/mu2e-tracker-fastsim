"""Saved nested rescore refuses incomplete trial rectangles."""
import json

import pytest

from scripts.secondary_bank_summary import run


def test_saved_rectangle_and_nonoverwrite(tmp_path):
    rows = [dict(bank=b, draw=d, metrics={"count": {"mean": b + d}}) for b in range(3) for d in range(2)]
    (tmp_path / "complete.json").write_text(json.dumps(dict(scope="test", replicas=rows)))
    result = run(tmp_path, tmp_path / "result.json")
    assert result["variance"]["count"]["bank_replicas"] == 3
    with pytest.raises(FileExistsError):
        run(tmp_path, tmp_path / "result.json")
    (tmp_path / "complete.json").write_text(json.dumps(dict(scope="test", replicas=rows[:-1])))
    with pytest.raises(ValueError, match="rectangle"):
        run(tmp_path, tmp_path / "invalid.json")
