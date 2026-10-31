# Native seed audit and first component probes — 2026-10-28

## Scope and outcome

GPVM02 became idle after its reconstruction finished. The existing capture was reused: no build, native recapture,
cleanup, baseline campaign, fitter change or default promotion.

**The seed operand reconstruction now has native-data evidence. Population acceptance/fidelity does not improve
by implication.** Original measured acceptance 95.594% and combined-seed momentum fidelity 89.808% remain the
latest population results, below the unchanged 99.5% / 99% gates.

All results live under `$M2FS_DATA/trk_f1native_r2_20261027/`.

## Independent seed audit

`seed_contract_v1.json` completed/passed/hash-stable true, **exact false**. Its original generator/audit and
preparation snapshot are now receipt-bound and remain unchanged.

| Quantity | Population | Result |
|---|---:|---|
| Generated KL seed parameters | 2,319 seeds / 13,914 components | Exact |
| Momentum from float32 normalization/scaling | 2,319 seeds | Exact; none exactly 2000 MeV |
| Ordered source membership, hit/crossing straw IDs and partners | 43,794 initial hits | Exact |
| Seed covariance, original double-square generator | 2,319 matrices | Tolerance pass; all inexact, max difference 1.7695128869121834e-10 |
| ParameterHit q/C/mask/index | 2,319 seeds | Exact |
| All-seed range and constraint time | 2,319 seeds | Exact, **conditioned on captured hit/crossing times** |
| Independently generated construction q/range | 147 targets | Exact |
| Independently generated hit/tube CAs, fixed offsets and effects | 2,538 constructors | Exact; zero moves/caps/exclusions |
| Independently generated fit range and constraint time | 147 targets | Exact |

Native seed momentum minus 2000 MeV has min/max **-0.00028441657559596933 / +0.0002749878653958149**.
The float32 normalization explanation is confirmed for this sample, not merely hypothesized. Native mass,
raw geometry and source membership remain supplied; LineFinder FitParams/MinuitParams equivalence is specific
to the frozen producer. These checks are not autonomous seeding or fitting.

Measured audit: **25.29 s**, maximum RSS **1,491,428 KiB**. Its 232 tests also pass remotely (**5.76 s**).
The receipt watches 728 paths. All tolerance comparisons retain rtol=1e-5 / atol=1e-7, with exactness separate.

## Supplemental exact seed covariance

The original discrepancy prompted inspection of `Offline/Mu2eKinKal/inc/KKFitSettings.hh:116`:
`SeedErrors` is **fhicl::Sequence<float>**. `KinematicLineFitTrace_module.cc:180–181` multiplies two float
operands before storing the result in the double matrix. The original generator instead squared double literals.

New `scripts/f1_seed_covariance_audit.py` implements only that source-typed arithmetic, in a separate audit:

```text
errors32 = float32([15, .05, 10, .02, 2, 1])
diagonal = float64(errors32 * errors32)
         = [225, .002500000176951289, 100, .00039999998989515007, 4, 1]
```

`seed_covariance_v1.json` is **completed/passed/exact/hash-stable true**: all **83,484 seed covariance
components** and **5,292 initial-construction covariance components** match exactly. No native matrix supplies
the generated values. The 1.7695e-10 and -1.0105e-11 differences from double squaring are reproduced explicitly.
The inspected header hash matches the earlier capture-area header already bound by the original receipt;
this does not create additional historical binary provenance.

The original `seed_contract_v1.json` is **not rewritten as exact**. Read it together with this supplemental
receipt. The old helper remains double-square and must not silently be promoted as an exact native covariance
constructor. Future integration should use a new version and retain both receipts.

**23 new tests pass locally and remotely**, covering independently specified float products, wrong promotion,
full-key joins/census, corruption with exact-vs-tolerance distinctions, protected/exclusive outputs, provenance,
and hash drift. No separate reviewer/native binary test is claimed. Supplemental audit: **3.70 s**, max RSS
**171,920 KiB**, 731 watched paths; remote tests **0.59 s**.

## Four single-target component probes

After pinning the actual seed receipt SHA, the unchanged `f1_seed_arm_probe.py` ran ordinary, explicit baseline
control, q-only, C-only, range-only and full substitution, separately in own/fixed modes. All four probes:

- completed with controls/hash stability true;
- matched **all 50 returned leaves exactly** for ordinary versus explicit baseline control before ablations;
- retained six complete NPZ outputs and strict manifests;
- had **no nonfinite output leaves** in any arm;
- used CPU/x64, JAX 0.11.2, NumPy 2.5.2, full stock schedule, unchanged fitter/defaults and identical explicit
  native constraint time across arms.

| Full key | Mode | Native FitOK | F1 good, all six arms | Status, all arms | Iterations, all arms | Range-only versus ordinary |
|---|---|---|---|---:|---:|---|
| 1400:0:3700:0 | Own | No | Yes | 0 | 27 | Exact complete tree |
| 1400:0:3700:0 | Fixed | No | No | 2 | 1 | Only initial owner range differs |
| 1400:0:7033:0 | Own | Yes | Yes | 0 | 28 | Exact complete tree |
| 1400:0:7033:0 | Fixed | Yes | Yes | 0 | 26 | Exact complete tree |

Event3700 has 19 hits/143 candidate crossings; event7033 has 20 hits/23 crossings. Both belong to the captured
147-target set and the 2,315 eligible input population. Event3700 fixed-state is deliberately preserved as a
**native-rejected diagnostic**, not included by implication in stock-good fidelity denominators. q-only, C-only
and full substitution change numerical leaves but do not change these targets' reported good/status/iteration
outcomes. This is not a measured fidelity improvement, momentum-only isolation or population causal result.
q still couples prior means and momentum-constraint target; C couples prior uncertainty/material operands.

The rejected fixed-state case confirms why range-only is not universally complete-tree identical: retained
initial owner bookkeeping differs even when fitted numbers/status do not. No range semantics are discarded.

Measured wall times for 3700 own/fixed and7033 own/fixed: **77.75 / 59.70 / 62.47 / 60.70 s**. No scaling estimate.
Every probe manifest watches737paths. Frozen full-population baselines remain authoritative and were not rerun.

## Immutable evidence

| Artifact | SHA-256 |
|---|---|
| seed_contract_v1.json | c1c7c19c7fc5cabdd85cd195ffa6973998a478fd9444cabd66882f3e677fa633 |
| seed_covariance_v1.json | e3b649e3b135bd68131b25c4faaaa2b14641a2c07c664ce1defa04084af4d740 |
| seed_probe_3700_own_v1/manifest.json | 53f7e5c9444e3e68bfd423a0630dbd71e864379bdb832e80e0379f7859d7841d |
| seed_probe_3700_fixed_v1/manifest.json | 4bc85f566bf9542936a6b18dbb245876523b14fc4645d1339dadb459de363dde |
| seed_probe_7033_own_v1/manifest.json | b564139b1aa80dffc024d69af518a06e8b1c67f8ebe359633024508583d639f0 |
| seed_probe_7033_fixed_v1/manifest.json | 100989227f45fb8a156a940296541369fe19f6b1ea502e34d1b59594b38b9d57 |
| seed_contract_receipts_v1.sha256 (9 entries) | d9eb1a73224c47560310f251af8b3b2457026f40305ed54fc3951ba1a5c13419 |
| seed_component_receipts_v1.sha256 (43 entries) | 097bdca2318dc8b5cce6368c5ed6bbe42be723132dfd8854042bf16194126ae1 |
| scripts/f1_seed_covariance_audit.py | 4e1bf6804a8c1a8488582b03f37264012cd65bdc7cc529c741d57e8fa60df889 |
| tests/test_f1_seed_covariance_audit.py | 83253708130ab36ac68d421d9b482a618a7620d4d65c8d0da2ed38a9c3832588 |

Both seals verified. Source/test snapshots and all logs are preserved. Fitter SHA remains
`988976b64b34f36b9021226b0088b364cc1c9b2b214158337c6c834e62b6e42a`; seed/arm/helper hashes match their original
snapshots. New files are unstaged. No git writes, build or cleanup.

## Next work

Implement the full-population component driver/comparator with preserved baseline compatibility controls,
complete eligible/stock-good/both-good and rejected/nonfinite denominators. Separate momentum from the other
q coordinates before attributing an effect to normalization. Keep exact covariance arithmetic in a new version,
not an edit to the original receipt-bound generator. Remaining state/CAD/activity/material and strict algebra/
actual-F1 decision closure are still open; the missing7gains/24losses still limit captured-history causal coverage.
