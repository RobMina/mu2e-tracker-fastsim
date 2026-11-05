"""Closed cosmic candidate stage history; fixed original event exposure, no physics inference."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from mu2efastsim import tables
from trkfastsim.validation import cecontract as cc, cehistory as ch, fieldpathaudit as pa

KEY = tables.KEY


def finished_log(path):
    text = Path(path).read_text()
    counts = re.findall(r"TrigReport Events total\s*=\s*(\d+)\s+passed\s*=\s*(\d+)\s+failed\s*=\s*(\d+)", text)
    if len(counts) != 1:
        raise ValueError("require exactly one finished event total: "+str(path))
    total, passed, failed = map(int, counts[0])
    if total != passed+failed or total <= 0:
        raise ValueError("invalid/empty finished event total")
    return dict(total=total, passed=passed, failed=failed)


def injection_filter_events(steps):
    """Strict-positive filter hypothesis using XYZVectorF component and norm arithmetic.

    Historical binary arithmetic is not attested; actual full FileIndex agreement
    is required independently and all momentum-boundary candidates are reported.
    """
    eion = steps.eion.to_numpy(dtype=np.float32)
    momentum = steps[["px","py","pz"]].to_numpy(dtype=np.float32)
    magnitude = np.sqrt(momentum[:,0]**2 + momentum[:,1]**2 + momentum[:,2]**2)
    good = (eion > 0.) & (magnitude > 0.) & (magnitude < 1e9)
    return cc.unique(steps.loc[good,KEY].drop_duplicates(),"stock-filter-positive step events")


def diagnose(source_table, candidate_table, stage_dir, outdir):
    """Saved-only filter-hypothesis inventory, independently checked against full FileIndex identities."""
    source_table,candidate_table,stage_dir,outdir=map(Path,(source_table,candidate_table,stage_dir,outdir))
    if outdir.exists():
        raise FileExistsError(outdir)
    source=tables.read(source_table,"particles",KEY).drop_duplicates()
    steps=tables.read(candidate_table,"sgsteps")
    d=cc.unique(source,"original source")
    allkeys=cc.subset(steps[KEY].drop_duplicates(),d,"all step-bearing events")
    filtered=cc.subset(injection_filter_events(steps),d,"positive-filter events")
    injected,trees=ch.load_art(stage_dir/"inj.art",byteswap=True)
    injected=cc.subset(injected,d,"injected diagnostic events")
    for name,frame in (("any_steps",allkeys),("filter_positive",filtered),("injected",injected)):
        d[name]=pd.MultiIndex.from_frame(d[KEY]).isin(pd.MultiIndex.from_frame(frame[KEY]))
    mismatch=d[d.filter_positive != d.injected].copy()
    excluded=d[d.any_steps & ~d.injected].copy()
    excluded_steps=steps.merge(excluded[KEY],on=KEY,validate="many_to_one")
    outdir.mkdir()
    d.to_csv(outdir/"events.csv.gz",index=False)
    excluded_steps.to_csv(outdir/"excluded_steps.csv.gz",index=False)
    mismatch.to_csv(outdir/"mismatched_events.csv.gz",index=False)
    result=dict(source_events=len(d),step_events=len(allkeys),filter_events=len(filtered),injected_events=len(injected),
                exact_filter_injection_mismatches=len(mismatch),any_steps_not_injected=len(excluded),
                injected_without_steps=int((d.injected & ~d.any_steps).sum()),
                excluded_step_rows=len(excluded_steps),excluded_zero_eion_rows=int((excluded_steps.eion==0).sum()),
                excluded_nonpositive_eion_rows=int((excluded_steps.eion<=0).sum()),highest_cycle_trees=trees,
                full_filter_keys_sha256=cc.fingerprint(filtered),injected_keys_sha256=cc.fingerprint(injected),
                exact_full_set_agreement=cc.fingerprint(filtered)==cc.fingerprint(injected),
                zero_momentum_component_rows=int((steps[["px","py","pz"]].to_numpy(dtype=np.float32)==0).all(axis=1).sum()),
                scope="saved-only strict-positive-eion filter hypothesis; full-set agreement required, not historical binary attestation, changed cuts or physics closure")
    (outdir/"fieldwallfilter.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


def reconcile(source, particles, steps, injected, digitized, reco, ntuple, filtered=None):
    den = cc.unique(source[KEY].drop_duplicates(), "original dts events")
    ent = cc.unique(particles[KEY].drop_duplicates(), "entry events")
    cc.subset(ent, den, "entry events")
    gas = cc.subset(steps[KEY].drop_duplicates(), ent, "step events")
    inj = cc.subset(injected, den, "injected")
    dig = cc.subset(digitized, den, "digitized")
    rec = cc.subset(reco, dig, "reco")
    nts = cc.subset(ntuple, den, "ntuple")
    accepted = gas if filtered is None else cc.subset(filtered,gas,"positive-filter events")
    for a,b,label in ((accepted,inj,"step/injection"),(inj,dig,"injection/digitization"),(rec,nts,"selected reco/NT")):
        if cc.fingerprint(a) != cc.fingerprint(b):
            raise ValueError("stage identity mismatch: "+label)
    out = den.copy()
    index = pd.MultiIndex.from_frame(out[KEY])
    for name, frame in (("entry",ent),("steps",gas),("stock_filter_positive",accepted),("injected",inj),
                         ("digitized",dig),("reco_selected",rec),("ntuple",nts)):
        out[name] = index.isin(pd.MultiIndex.from_frame(frame[KEY]))
    return out


def run(source_table, candidate_table, stage_dir, legacy_dir, outdir):
    source_table, candidate_table, stage_dir, legacy_dir, outdir = map(
        lambda x:Path(x).resolve(), (source_table,candidate_table,stage_dir,legacy_dir,outdir))
    if outdir.exists():
        raise FileExistsError(outdir)
    paths = dict(source_table=source_table,candidate_table=candidate_table)
    stage_paths = {name:stage_dir/file for name,file in (("injected","inj.art"),("digitized","dig.art"),("reco","mcs-CH.art"))}
    paths.update(stage_paths)
    paths["ntuple"] = stage_dir/"nts.001430_00000000.root"
    paths.update({"log/"+stage:stage_dir/(stage+".log") for stage in ("inject","digi","reco","ntuple")})
    for name in ("fast","g4fp"):
        directory = legacy_dir/("art3v2_"+name+"_unbiased")
        paths["legacy/"+name+"/ntuple"] = directory/"nts.001430_00000000.root"
        paths["legacy/"+name+"/ntuple_log"] = directory/"001430_00000000"/"ntuple.log"
    expected_source = "60c88e2e4f7fb9bd9a363bc63aff5dca8322794d51264c1592cda89fd4f9593c"
    if pa.sha256(source_table) != expected_source:
        raise ValueError("source table differs from original closed dumper receipt")
    original_art = legacy_dir.parent/"inputs/cosmiccryall_ac/dts.mu2e.CosmicCRYAll.MDC2025ac.001430_00000000.art"
    paths["original_art"] = original_art
    if pa.sha256(original_art) != "97de66db166178604092a6fd19c411ac558301ae8b7e6fbd993d80c2e8484a06":
        raise ValueError("original DTS differs from source audit receipt")
    before = pa.receipts(paths)
    source = tables.read(source_table,"particles",KEY)
    if len(source[KEY].drop_duplicates()) != 52037:
        raise ValueError("original fixed source exposure must contain 52037 dts events")
    original_keys, original_trees = ch.load_art(original_art,byteswap=True)
    if cc.fingerprint(source[KEY].drop_duplicates()) != cc.fingerprint(original_keys):
        raise ValueError("source dumper event keys differ from original DTS")
    particles = tables.read(candidate_table,"particles")
    steps = tables.read(candidate_table,"sgsteps")
    frames, trees = {}, {}
    for name,path in stage_paths.items():
        frames[name], trees[name] = ch.load_art(path,byteswap=True)
    frames["ntuple"], trees["ntuple"] = ch.load_ntuple(paths["ntuple"])
    filtered = injection_filter_events(steps)
    ledger = reconcile(source,particles,steps,frames["injected"],frames["digitized"],frames["reco"],frames["ntuple"],filtered)
    logs = {stage:finished_log(paths["log/"+stage]) for stage in ("inject","digi","reco","ntuple")}
    expected = dict(inject=len(ledger),digi=int(ledger.injected.sum()),reco=int(ledger.digitized.sum()),ntuple=len(frames["ntuple"]))
    if any(logs[k]["total"] != v for k,v in expected.items()):
        raise ValueError("finished log processing count != actual stage identities")
    if logs["inject"]["passed"] != len(frames["injected"]) or logs["digi"]["passed"] != len(frames["digitized"]):
        raise ValueError("finished log pass count != actual selected stage identities")
    legacy = {}
    for name in ("fast","g4fp"):
        npath,lpath = paths["legacy/"+name+"/ntuple"],paths["legacy/"+name+"/ntuple_log"]
        f,t = ch.load_ntuple(npath)
        log = finished_log(lpath)
        if len(f) != log["total"]:
            raise ValueError("retained NT count != finished log: "+name)
        legacy[name] = dict(entries=len(f),good_events=int((f.ntrack>0).sum()),finished_log=log,trees=t)
    if pa.receipts(paths) != before:
        raise ValueError("closed stage input changed during history read")
    outdir.mkdir()
    ledger.to_csv(outdir/"events.csv.gz",index=False)
    result = dict(passed=True,source_events=len(ledger),entry_particles=len(particles),
                  stages={k:int(ledger[k].sum()) for k in ledger if k not in KEY},
                  source_identity_sha256=cc.fingerprint(ledger[KEY]),original_art_trees=original_trees,
                  highest_cycle_trees=trees,finished_logs=logs,retained_baselines=legacy,inputs=before,
                  identity_sha256={k:cc.fingerprint(ledger.loc[ledger[k],KEY]) for k in ledger if k not in KEY},
                  scope="fixed original selected-dts exposure and actual stage identity/filtering only; not generated cosmics or physics closure",
                  reco_filter_note="stock TrigReport passed may differ from selected CentralHelix output; actual rec/NT sets must agree")
    (outdir/"fieldwallhistory.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("source-table","candidate-table","stage-dir","outdir"):
        ap.add_argument("--"+name,required=True)
    ap.add_argument("--legacy-dir")
    ap.add_argument("--diagnose-filter",action="store_true")
    args=vars(ap.parse_args(argv))
    diagnostic=args.pop("diagnose_filter")
    if diagnostic:
        args.pop("legacy_dir")
        result=diagnose(**args)
    else:
        if not args["legacy_dir"]:
            ap.error("--legacy-dir required for full history")
        result=run(**args)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
