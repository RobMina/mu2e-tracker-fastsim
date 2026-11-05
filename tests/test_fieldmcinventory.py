"""Inventory nesting and missing MC products without claiming exact physics associations."""
import awkward as ak

from trkfastsim.validation import fieldmcinventory as inv


class Branch:
    def __init__(self, data):
        self.data = data

    def array(self, library):
        assert library == 'ak'
        return self.data


class Tree(dict):
    object_path = '/EventNtuple/ntuple;2'
    num_entries = 2


class Root(dict):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_actual_fields_and_nested_records(tmp_path, monkeypatch):
    path = tmp_path / 'ntuple.root'
    path.write_bytes(b'closed fixture')
    tree = Tree(trk=Branch(ak.Array([[{'status': 1}], []])),
                trkmcsim=Branch(ak.Array([[[{'valid': True, 'rank': 0, 'id': 7}]], []])),
                trksegsmc=Branch(ak.Array([[[]], []])))
    monkeypatch.setattr(inv.uproot, 'open', lambda _: Root({'EventNtuple/ntuple': tree}))
    result = inv.inventory(path)
    assert result['entries'] == 2 and result['tree'].endswith(';2')
    assert result['branches']['trk']['track_records'] == 1
    assert result['branches']['trkmcsim']['nested_records'] == 1
    assert result['branches']['trkmcsim']['fields'] == ['valid', 'rank', 'id']
    assert result['branches']['trksegsmc']['nested_records'] == 0
    assert result['branches']['trkhitsmc'] == {'present': False}


def test_record_of_jagged_leaves_uses_a_leaf_for_counts(tmp_path, monkeypatch):
    path = tmp_path / 'ntuple.root'
    path.write_bytes(b'closed record fixture')
    tree = Tree(trk=Branch(ak.Array({'status': [[1, 2], []]})),
                trkmcsim=Branch(ak.Array({'id': [[[7], []], []], 'rank': [[[0], []], []]})))
    monkeypatch.setattr(inv.uproot, 'open', lambda _: Root({'EventNtuple/ntuple': tree}))
    result = inv.inventory(path)
    assert result['branches']['trk']['track_records'] == 2
    assert result['branches']['trkmcsim']['nested_records'] == 1
