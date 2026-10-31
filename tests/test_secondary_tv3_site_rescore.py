"""Site relocation preserves payload and does not call it current transport validation."""
import json
from pathlib import Path

import pytest

from scripts import secondary_tv3_site_rescore as S


def test_relocate_requires_containment_and_all_payloads(tmp_path):
    root = tmp_path / 'stage'
    root.mkdir()
    (root / 'input.root').touch()
    assert S.relocate({'a': ['/old/input.root']}, '/old', root) == {'a': [str(root / 'input.root')]}
    for name in ['/old/missing', '/other/input', '/old/../escape']:
        with pytest.raises(ValueError):
            S.relocate({'a': [name]}, '/old', root)
    outside = tmp_path / 'outside'
    outside.touch()
    (root / 'link').symlink_to(outside)
    with pytest.raises(ValueError, match='escaped'):
        S.relocate({'a': ['/old/link']}, '/old', root)


def test_rescore_ignores_only_relocation_and_hashes_payloads(tmp_path, monkeypatch):
    root = tmp_path / 'stage'
    root.mkdir()
    (root / 'input.root').write_bytes(b'synthetic')
    kinds = ('ntuple_fast', 'ntuple_g4', 'steps_fast', 'steps_g4', 'hits_fast', 'hits_g4')
    saved = dict(paired=True, input_files={k: ['/old/input.root'] for k in kinds}, value=1)
    path = tmp_path / 'comparison.json'
    path.write_text(json.dumps(saved))
    original = path.read_bytes()
    def run(inputs, outdir, **kwargs):
        assert inputs == [str(root / 'input.root')]
        assert kwargs['paired']
        Path(outdir).mkdir()
        new = dict(saved, input_files={k: [str(root / 'input.root')] for k in kinds}, value=2)
        (Path(outdir) / 'tv2compare.json').write_text(json.dumps(new))
    monkeypatch.setattr(S.TV, 'run', run)
    receipt = tmp_path / 'receipt.txt'
    receipt.write_text(S.digest(root / 'input.root') + '  /old/input.root\n')
    result = S.run(path, '/old', root, tmp_path / 'out', receipt)
    assert result['source_receipt_payloads_checked'] == 1
    assert result['changed_fields_excluding_relocation'] == ['value']
    assert len(result['inputs']) == 1
    assert result['inputs'][str(root / 'input.root')] == S.digest(root / 'input.root')
    assert path.read_bytes() == original
    assert 'historical' in result['scope']
    with pytest.raises(FileExistsError):
        S.run(path, '/old', root, tmp_path / 'out')
    receipt.write_text('0' * 64 + '  /old/input.root\n')
    with pytest.raises(ValueError, match='differs from source'):
        S.run(path, '/old', root, tmp_path / 'bad', receipt)
    assert not (tmp_path / 'bad/complete.json').exists()
