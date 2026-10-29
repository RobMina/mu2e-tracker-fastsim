"""Static and isolated quota-function tests; never execute the native job driver."""
from pathlib import Path
import re
import subprocess

import pytest

DRIVER = Path(__file__).resolve().parents[1] / "scripts/run_f1_crossing_full_pair.sh"
TEXT = DRIVER.read_text()


def test_shell_syntax_and_fixed_native_count():
    subprocess.run(["bash", "-n", str(DRIVER)], check=True)
    assert "-n 3000" in TEXT and "events=3000" in TEXT
    assert "N=${" not in TEXT
    assert "f1native-crossing-full-20261028" in TEXT
    assert 'taskset -c "$CPU" mu2e' in TEXT
    assert "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1" in TEXT


def test_outputs_exclusive_and_canonical_before_creation():
    canonical = TEXT.index('OUT=$(realpath -m "$OUT")')
    guard = TEXT.index('[ ! -e "$OUT" ]')
    create = TEXT.index('mkdir "$OUT"')
    assert canonical < guard < create
    assert '"$(realpath -e "$F1N")/"*' in TEXT
    assert not re.search(r"\b(rm|rsync|scp|scons|make|cmake)\b", TEXT)
    assert "muse build" not in TEXT


def test_frozen_receipts_and_configuration_before_native():
    for sha in ("ce3762ab9a5610e5838a5939ccff47c67d362812a41ddb1428eca13cc269cd5d",
                "9b9983124f8df4f48b253158f414f7aa9712d03d1eb0b22e250ea47810f12eb1",
                "101623117e0f245ad02982cd01ff88a15b7411469319199fee906ae22232c1da",
                "b97804739a88d739d6fcda45954edd4a88b322e67d2a58929b333971f4c7beb6",
                "cc3441c37613151767bdb9bcbb24a8e7d94813d2859c095a87a15ee4afe85439"):
        assert sha in TEXT
    native = TEXT.index('/usr/bin/time -v taskset')
    assert TEXT.index('cmp "$OUT/$arm.normalized.fcl"') < native
    assert TEXT.index('ldd "$PLUGIN"') < native
    assert TEXT.index('source; do') < native
    assert '"$SMOKE/manifest.txt"' in TEXT


@pytest.mark.parametrize("free,success", [(8 * 1024**3, True), (8 * 1024**3 + 1, True),
    (8 * 1024**3 - 1, False), (4 * 1024**3, False), (0, False), (-1, False)])
def test_quota_function_exact_boundary(free, success):
    function = TEXT.split("quota_gate() {", 1)[1].split("\n}\n", 1)[0]
    limit, used = 150 * 1024**3, 150 * 1024**3 - free
    script = f'''set -e -o pipefail
DATA=/synthetic
ALLOWANCE=$((4*1024*1024*1024))
RESERVE=$((4*1024*1024*1024))
getfattr() {{ case "$*" in *max_bytes*) printf '%s' {limit};; *) printf '%s' {used};; esac; }}
quota_gate() {{{function}
}}
quota_gate
'''
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert (result.returncode == 0) is success


def test_space_policy_uses_measured_bytes_not_runtime():
    proxy = (15152564 * 147 + 1) // 2 + 17277322 + 117244629
    assert proxy == 1248235405
    assert 4 * 1024**3 >= 2 * proxy
    assert 'stat -c %s "$SMOKE/capture.root"' in TEXT
    assert "no runtime extrapolation" in TEXT
    assert TEXT.count("  quota_gate\n") == 1


@pytest.mark.parametrize("failing", ["control", "capture"])
def test_each_predecessor_stat_failure_stops(failing):
    fragment = TEXT.split('CONTROL_BYTES=$(stat', 1)[1].split('ALLOWANCE=', 1)[0]
    fragment = 'CONTROL_BYTES=$(stat' + fragment
    script = f'''set -e -o pipefail
REFERENCE=/synthetic
stat() {{ case "$*" in *{failing}.root*) return 7;; *) printf '123';; esac; }}
{fragment}
echo BAD_CONTINUATION
'''
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode == 7
    assert "BAD_CONTINUATION" not in result.stdout
