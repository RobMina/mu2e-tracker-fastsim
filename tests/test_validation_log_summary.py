"""Full-suite aggregation keeps failures, missing files and trailing unfinished tests explicit."""
import pytest

from scripts.validation_log_summary import summarize


def test_complete_counts_and_legacy_expected_failures():
    result = summarize('==== tests/a.py\n3 passed, 2 xfailed in 1.0s\n==== tests/b.py\n4 passed, 1 skipped in 2s',
                       ['tests/a.py', 'tests/b.py'])
    assert result['complete'] and result['no_unexpected_failures']
    assert result['totals']['passed'] == 7 and result['totals']['xfailed'] == 2


def test_failures_missing_and_unfinished_never_hidden():
    result = summarize('==== tests/a.py\n2 failed, 1 passed in 1.0s\n==== tests/b.py\n...',
                       ['tests/a.py', 'tests/b.py', 'tests/c.py'])
    assert not result['complete'] and not result['no_unexpected_failures']
    assert result['missing'] == ['tests/c.py'] and result['unfinished'] == ['tests/b.py']
    with pytest.raises(ValueError, match='duplicate'):
        summarize('==== tests/a.py\n1 passed in 1s\n==== tests/a.py', ['tests/a.py'])
