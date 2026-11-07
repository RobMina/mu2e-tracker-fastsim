"""External process timing parsing for the implemented benchmark routes."""
import pytest
from trkfastsim.validation.entry_benchmark import process_time


@pytest.mark.parametrize("clock,seconds", [("1:27.37", 87.37), ("12.34", 12.34), ("1:02:03.4", 3723.4)])
def test_time_formats(tmp_path, clock, seconds):
    path = tmp_path / "process.time"
    path.write_text(f"User time (seconds): 3.1\nSystem time (seconds): 0.2\n"
                    f"Elapsed (wall clock) time (h:mm:ss or m:ss): {clock}\n"
                    "Maximum resident set size (kbytes): 2000\nExit status: 0\n")
    result = process_time(path)
    assert result["wall_s"] == pytest.approx(seconds)
    path.write_text(path.read_text().replace("Exit status: 0", "Exit status: 65"))
    with pytest.raises(ValueError, match="unsuccessful"):
        process_time(path)
