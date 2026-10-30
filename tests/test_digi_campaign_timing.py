"""Campaign timings require complete measured passes and matched populations."""
import json
import pytest
import digi_campaign_timing as campaign


def test_campaign_rejects_partial_and_sums_measured_rows(tmp_path):
    rows = [dict(repeat=repeat, file_index=i, events=20, steps=100, digis=50,
                 seconds=float(1 + repeat), stats={"digitize_s": .5, "jax_buckets": 2})
            for repeat in range(3) for i in range(10)]
    path = tmp_path / "timing.json"
    path.write_text(json.dumps(dict(reproducible=True, runs=rows)))
    got = campaign.passes(tmp_path)
    assert [r["wall_s"] for r in got] == [10., 20., 30.]
    assert got[0]["events"] == 200 and got[0]["electronics_calls"] == 20
    path.write_text(json.dumps(dict(reproducible=True, runs=rows[:-1])))
    with pytest.raises(ValueError, match="ten files"):
        campaign.passes(tmp_path)
