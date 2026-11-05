"""Historical stock cosmic filter receipts retain missing entries and multiple particles."""
import numpy as np
import pandas as pd
import pytest

from trkfastsim.validation import fieldwallhistory as wh


def ids(n=6):
    return pd.DataFrame(dict(run=np.full(n, 2**63+1,dtype=np.uint64), subrun=np.zeros(n,dtype=np.uint64),
                             event=np.arange(n,dtype=np.uint64)+np.uint64(2**63+3)))


def test_multiparticle_entries_zero_steps_allsourceevents_and_exact_stagekeys():
    source=ids()
    parts=pd.concat([source.iloc[:4],source.iloc[[0]]],ignore_index=True).assign(key=[1,1,1,1,2])
    gas=source.iloc[:3]
    rec=source.iloc[:1]
    out=wh.reconcile(source,parts,gas,gas,gas,rec,rec)
    assert len(out)==6 and out.event.dtype==np.uint64
    assert out.entry.sum()==4 and out.steps.sum()==3 and out.ntuple.sum()==1
    assert out.loc[3,'entry'] and not out.loc[3,'steps']
    assert not out.loc[4,'entry']


@pytest.mark.parametrize('which',['inject','digi','reco','ntuple'])
def test_stage_mismatch_fails_closed(which):
    source=ids()
    frames=[source.iloc[:3].copy(),source.iloc[:3].copy(),source.iloc[:2].copy(),source.iloc[:2].copy()]
    index=['inject','digi','reco','ntuple'].index(which)
    frames[index]=frames[index].iloc[:-1]
    with pytest.raises(ValueError,match='mismatch|outside'):
        wh.reconcile(source,source,source.iloc[:3],*frames)


def test_finished_log_positive_unique_conservation(tmp_path):
    p=tmp_path/'ntuple.log'
    p.write_text('TrigReport Events total = 17 passed = 17 failed = 0\n')
    assert wh.finished_log(p)==dict(total=17,passed=17,failed=0)
    for text in ('',p.read_text()*2,'TrigReport Events total = 17 passed = 16 failed = 0',
                 'TrigReport Events total = 0 passed = 0 failed = 0'):
        p.write_text(text)
        with pytest.raises(ValueError):
            wh.finished_log(p)


def test_stock_strict_positive_filter_keeps_zero_deposition_in_exposure_not_injection():
    source=ids()
    steps=source.iloc[:4].assign(eion=[1.,0.,np.nextafter(0.,1.),1.],px=[1.,1.,1.,1e9],py=0.,pz=0.)
    accepted=wh.injection_filter_events(steps)
    assert len(accepted)==1 and accepted.event.iloc[0]==source.event.iloc[0]
    out=wh.reconcile(source,source,steps,accepted,accepted,accepted,accepted,accepted)
    assert out.steps.sum()==4 and out.stock_filter_positive.sum()==1
    assert out.injected.sum()==1 and len(out)==6
    assert (out.steps & ~out.injected).sum()==3


def test_equal_count_wrong_identity_is_not_stage_closure():
    source=ids()
    left,right=source.iloc[:3],source.iloc[1:4]
    with pytest.raises(ValueError,match='identity mismatch'):
        wh.reconcile(source,source,left,right,right,right,right,left)


def test_positive_filter_float32_momentum_norm_zero_and_multiple_particles():
    base=ids()
    steps=base.iloc[:3].assign(eion=1.,px=[0.,1e-30,100.],py=0.,pz=0.)
    extra=steps.iloc[[0]].assign(px=200.)
    mixed=pd.concat([steps,extra],ignore_index=True)
    selected=wh.injection_filter_events(mixed)
    np.testing.assert_array_equal(selected.event,base.iloc[[0,2]].event)


def test_diagnostic_rejects_foreign_injected_keys_not_hidden_by_denominator(tmp_path,monkeypatch):
    source=ids()
    steps=source.iloc[:3].assign(eion=1.,px=100.,py=0.,pz=0.)
    monkeypatch.setattr(wh.tables,'read',lambda path,tree,columns=None: source if tree=='particles' else steps)
    foreign=source.iloc[:3].copy()
    foreign.loc[0,'event']+=np.uint64(100)
    monkeypatch.setattr(wh.ch,'load_art',lambda *a,**k:(foreign,{}))
    with pytest.raises(ValueError,match='outside'):
        wh.diagnose(tmp_path/'source',tmp_path/'candidate',tmp_path/'stages',tmp_path/'out')
