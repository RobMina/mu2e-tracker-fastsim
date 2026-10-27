"""Render a completed F1 closure JSON report as compact Markdown; no ROOT I/O or fitting."""
import argparse
import json
from pathlib import Path


def render(report):
    lines = ["# F1 closure diagnostic results", "", report["scope"], "",
             "Time-model arms are stock-inspired approximations, not exact native closure.", "",
             "## Own-state yield on all eligible seeds", "",
             "| Arm | F1 good | Gains | Losses | Net / stock (%) | Acceptance agreement (%) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, arm in report["arms"].items():
        if "yield" not in arm:
            continue
        y = arm["yield"]
        net = f"{100*y['net_rel']:+.3f} ± {100*y['net_rel_se']:.3f}" if y["net_rel"] is not None else "undefined"
        lines.append(f"| {name} | {y['f1_good']} | {y['gains']} | {y['losses']} | {net} | {100*y['acceptance_agreement']:.3f} |")
    lines += ["", "## Fixed-state fidelity (both accepted)", "",
              "Fractions within 0.1 stock sigma, percent; target 99% per parameter.", "",
              "| Arm | Both | d0 | phi0 | z0 | theta | t0 | Momentum |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, arm in report["arms"].items():
        if "fidelity" not in arm:
            continue
        f = arm["fidelity"]
        values = [f"{100*f[k]['frac_within_0p1']:.3f}" if f[k]["frac_within_0p1"] is not None else "undefined"
                  for k in ("d0", "phi0", "z0", "theta", "t0", "mom")]
        lines.append(f"| {name} | {f['n_tracks']} | " + " | ".join(values) + " |")
    lines += ["", "## Losses and numerical-validity diagnostics", ""]
    for name, arm in report["arms"].items():
        if "yield" not in arm:
            continue
        lines.append(f"- **{name}** losses: `{arm['loss_mechanism_f1']}`; gains: `{arm['gain_mechanism_stock']}`.")
        div = arm.get("divergence_diagnostic", {})
        if div:
            lines.append(f"  Front marginal momentum term over threshold: {div['front_marginal_momentum_over_threshold']} / "
                         f"{div['n_paramsdiverged_losses']} ParamsDiverged losses (not a joint-chi2 decomposition).")
        val = arm.get("numerical_validity")
        if val:
            p = val["populations"]["f1_good"]
            lines.append(f"  Final-state numerical validity: {p['n_valid']} / {p['n_tracks']} F1-good; acceptance unchanged.")
    lines += ["", "## Paired comparisons and outlier diagnostics", ""]
    for name in ("own_legacy", "own_stock", "own_stock_Lcons"):
        arm = report["arms"].get(name, {})
        for base in ("legacy", "candidate", "prior"):
            if "vs_" + base in arm:
                lines.append(f"- {name} versus saved {base}: `{arm['vs_' + base]}`")
    for key, value in report.get("cross", {}).items():
        if key.startswith("own_stock->"):
            lines.append(f"- {key}: `{value}`")
    for name in ("fixed_legacy", "fixed_stock", "fixed_stock_xstock"):
        f = report["arms"].get(name, {}).get("fidelity")
        if not f:
            continue
        lines += [f"- **{name}** geometry outliers: `{f.get('outlier_attribution')}`",
                  f"  Momentum difference [MeV/c] q16/50/84: `{f['mom']['difference_q16_50_84']}`; "
                  f"pull q16/50/84: `{f['mom']['q16_50_84']}`."]
        errors = {k: f[k]["err_frac_within_2pct"] for k in ("d0", "phi0", "z0", "theta", "t0", "mom")}
        lines.append(f"  Fractions with errors within 2%: `{errors}`.")
    lines += ["", "## Input event-count gate", "", "```json", json.dumps(report.get("inputs", {}), indent=2), "```", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report")
    p.add_argument("--out")
    a = p.parse_args()
    text = render(json.loads(Path(a.report).read_text()))
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
