"""Plot a saved, event-paired cosmic comparison without running transport or reconstruction."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["run", "subrun", "event"]
MOMENTUM_EDGES = np.array([0, 300, 500, 1000, 2000, 5000, 10000, 30000, np.inf])
MIN_SHAPE_ENTRIES = 10


def mean_error(values):
    values = np.asarray(values, float)
    return {"value": float(values.mean()) if len(values) else None,
            "error": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
            "events": len(values)}


def efficiency(fast, stock):
    fast, stock = np.asarray(fast, bool), np.asarray(stock, bool)
    difference = mean_error(100 * (fast.astype(float) - stock))
    ratio = None
    if stock.sum():
        value = fast.sum() / stock.sum()
        residual = fast.astype(float) - value * stock
        error = (np.sqrt(np.sum(residual**2) * len(stock) / (len(stock) - 1)) / stock.sum()
                 if len(stock) > 1 else None)
        ratio = {"value": float(value), "error": float(error) if error is not None else None}
    return dict(events=len(fast), fast=int(fast.sum()), stock=int(stock.sum()),
                fast_percent=mean_error(100 * fast), stock_percent=mean_error(100 * stock),
                difference_pp=difference, ratio=ratio,
                gained=int((fast & ~stock).sum()), lost=int((stock & ~fast).sum()))


def indicators(values, edges, selected):
    values = np.asarray(values, float)
    bins = np.searchsorted(edges, values, side="right") - 1
    bins = np.where(values == edges[-1], len(edges) - 2, bins)
    valid = selected & np.isfinite(values) & (bins >= 0) & (bins < len(edges) - 1)
    return (bins[:, None] == np.arange(len(edges) - 1)) & valid[:, None]


def paired_histogram(fast, stock, edges, fast_selected, stock_selected):
    """Event covariance gives paired ratios and a normalized-shape chi-square on its supported subspace."""
    a = indicators(fast, edges, fast_selected).astype(float)
    b = indicators(stock, edges, stock_selected).astype(float)
    ha, hb = a.sum(axis=0), b.sum(axis=0)
    ratio = np.divide(ha, hb, out=np.full_like(ha, np.nan), where=hb > 0)
    residual = a - np.nan_to_num(ratio) * b
    correction = len(a) / (len(a) - 1) if len(a) > 1 else 0
    ratio_error = np.divide(np.sqrt(correction * (residual**2).sum(axis=0)), hb,
                            out=np.full_like(ha, np.nan), where=hb > 0)
    result = dict(edges=np.asarray(edges).tolist(), fast=ha.tolist(), stock=hb.tolist(),
                  ratio=ratio.tolist(), ratio_error=ratio_error.tolist(),
                  fast_outside=int(np.sum(fast_selected) - ha.sum()),
                  stock_outside=int(np.sum(stock_selected) - hb.sum()))
    included = (ha >= MIN_SHAPE_ENTRIES) & (hb >= MIN_SHAPE_ENTRIES)
    shape = dict(chi2=None, ndf=0, included=included.tolist(), minimum_entries_per_arm=MIN_SHAPE_ENTRIES,
                 excluded_bins=int((~included).sum()), excluded_fast_entries=int(ha[~included].sum()),
                 excluded_stock_entries=int(hb[~included].sum()),
                 pulls=np.full(len(ha), np.nan).tolist(), covariance=np.zeros((len(ha), len(ha))).tolist(),
                 normalization="each arm within included bins only",
                 statistic="paired event covariance; pseudoinverse rank is ndf")
    if included.any():
        sa, sb = a[:, included], b[:, included]
        na, nb = ha[included].sum(), hb[included].sum()
        pa, pb = ha[included] / na, hb[included] / nb
        influence = (sa - sa.sum(axis=1)[:, None] * pa) / na
        influence -= (sb - sb.sum(axis=1)[:, None] * pb) / nb
        covariance = correction * influence.T @ influence
        delta = pa - pb
        error = np.sqrt(np.maximum(0, np.diag(covariance)))
        pulls = np.full(len(ha), np.nan)
        pulls[included] = np.divide(delta, error, out=np.full_like(delta, np.nan), where=error > 0)
        eig, vectors = np.linalg.eigh(covariance)
        keep = eig > (eig.max() * 1e-10 if eig.max() > 0 else 0)
        components = vectors[:, keep].T @ delta
        unsupported = float(np.linalg.norm(vectors[:, ~keep].T @ delta))
        chi2 = float(np.sum(components**2 / eig[keep])) if unsupported < 1e-10 else np.inf
        full_covariance = np.zeros((len(ha), len(ha)))
        full_covariance[np.ix_(included, included)] = covariance
        shape.update(chi2=chi2, ndf=int(keep.sum()), unsupported_difference_norm=unsupported,
                     pulls=pulls.tolist(), covariance=full_covariance.tolist())
    result["shape"] = shape
    return result


def compare(events):
    if events.duplicated(KEY).any() or len(events) < 2:
        raise ValueError("require distinct paired event keys and at least two events")
    good_a = events.legacy_ntrk.fillna(0).to_numpy() > 0
    good_b = events.g4_ntrk.fillna(0).to_numpy() > 0
    result = dict(scope="one explored development parent; no held-out confirmation",
                  configuration="saved legacy wall arm; historical solid gas branch and no radiative loss, not all current-main defaults",
                  selection="first track with status > 0; fixed saved G4 first-straw event denominator",
                  momentum="earliest in-bounds SID 0..4; reconstructed momentum, not truth-matched resolution",
                  efficiency=efficiency(good_a, good_b), efficiency_bins=[], histograms={})
    binned = np.isfinite(events.p_first) & (events.p_first >= MOMENTUM_EDGES[0]) & (events.p_first < MOMENTUM_EDGES[-1])
    result["efficiency_outside"] = dict(events=int((~binned).sum()),
                                        fast=int(good_a[~binned].sum()), stock=int(good_b[~binned].sum()),
                                        reason="nonfinite or outside efficiency-bin range; retained in total efficiency")
    for lo, hi in zip(MOMENTUM_EDGES[:-1], MOMENTUM_EDGES[1:]):
        selected = binned & (events.p_first >= lo) & (events.p_first < hi)
        result["efficiency_bins"].append(dict(lo_MeV=float(lo), hi_MeV=float(hi),
                                              **efficiency(good_a[selected], good_b[selected])))
    for name, edges in (("p_reco", np.geomspace(100, 1_000_000, 41)),
                        ("nactive", np.arange(0, 85, 5)), ("chi2ndf", np.linspace(0, 10, 41))):
        result["histograms"][name] = paired_histogram(events[f"legacy_{name}"], events[f"g4_{name}"], edges,
                                                       good_a, good_b)
    return result


def _shape_label(shape):
    value = shape["chi2"]
    return "Shape χ²/ndf: insufficient populated bins" if value is None else f"Shape χ²/ndf = {value:.1f}/{shape['ndf']}"


def plot(result, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#0072B2", "#D55E00")
    fig = plt.figure(figsize=(13, 9), layout="constrained")
    outer = fig.add_gridspec(2, 2)
    axes = []
    for cell in outer:
        inner = cell.subgridspec(2, 1, height_ratios=(3, 1), hspace=0.03)
        top = fig.add_subplot(inner[0])
        bottom = fig.add_subplot(inner[1], sharex=top)
        top.tick_params(labelbottom=False)
        axes.append((top, bottom))
    top, bottom = axes[0]
    bins = result["efficiency_bins"]
    x = np.arange(len(bins))
    labels = [f"{b['lo_MeV']/1000:g}–{b['hi_MeV']/1000:g}" if np.isfinite(b["hi_MeV"])
              else f"≥{b['lo_MeV']/1000:g}" for b in bins]
    for arm, color, offset, label in (("fast", colors[0], -0.07, "Fast: solid gas, no radiative loss, wall off"),
                                      ("stock", colors[1], 0.07, "G4 first-passage steps; both stock reco")):
        top.errorbar(x + offset, np.asarray([b[f"{arm}_percent"]["value"] for b in bins], float),
                     yerr=np.asarray([b[f"{arm}_percent"]["error"] for b in bins], float),
                     fmt="o", ms=4, color=color, label=label)
    bottom.errorbar(x, np.asarray([b["ratio"]["value"] if b["ratio"] else np.nan for b in bins], float),
                    yerr=np.asarray([b["ratio"]["error"] if b["ratio"] else np.nan for b in bins], float),
                    fmt="o", ms=4, color=colors[0])
    bottom.set_xticks(x, labels, rotation=25)
    bottom.set_xlabel("G4 event-first-straw momentum [GeV/c]")
    top.set_ylabel("Good-track efficiency [%]")
    top.legend(fontsize=8, loc="upper left")
    eff = result["efficiency"]
    diff = eff["difference_pp"]
    top.set_title(f"{eff['fast']:,} / {eff['stock']:,} good events of {eff['events']:,}; "
                  f"unbinned: {result['efficiency_outside']['events']}\n"
                  f"Fast − G4: {diff['value']:+.3f} ± {diff['error']:.3f} pp", fontsize=11)
    names = (("p_reco", "Reconstructed momentum [GeV/c]", 0.001),
             ("nactive", "Active hits / first good track", 1),
             ("chi2ndf", r"Track $\chi^2$/ndf", 1))
    for (top, bottom), (name, xlabel, scale) in zip(axes[1:], names):
        h = result["histograms"][name]
        edges = np.asarray(h["edges"]) * scale
        centers = (edges[1:] + edges[:-1]) / 2
        for arm, color in (("fast", colors[0]), ("stock", colors[1])):
            top.stairs(h[arm], edges, color=color, linewidth=1.5)
        bottom.errorbar(centers, h["ratio"], yerr=h["ratio_error"], fmt=".", color=colors[0], ms=4)
        top.set_ylabel("Good-track events / bin")
        if name == "p_reco":
            top.set_xscale("log")
        bottom.set_xlabel(xlabel)
        shape = h["shape"]
        top.set_title(f"{_shape_label(shape)}; ≥{MIN_SHAPE_ENTRIES}/arm\n"
                      f"Outside range: fast {h['fast_outside']}, G4 {h['stock_outside']}; "
                      f"low-count omitted: {shape['excluded_fast_entries']}/{shape['excluded_stock_entries']}", fontsize=10)
    for top, bottom in axes:
        top.grid(alpha=0.18)
        bottom.axhline(1, color="0.4", lw=0.8)
        bottom.set_ylabel("Fast/G4")
        bottom.grid(alpha=0.18)
    fig.suptitle("Field-on cosmic tracks — saved paired development sample", fontsize=15)
    fig.supxlabel("Paired event errors; historical physics, legacy wall. One explored parent, not held-out validation. "
                  "Momentum is not a common-surface resolution.", fontsize=9)
    fig.savefig(destination, dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for axis, (name, xlabel, scale) in zip(axes, names):
        h = result["histograms"][name]
        edges = np.asarray(h["edges"]) * scale
        centers = (edges[1:] + edges[:-1]) / 2
        supported = np.asarray(h["shape"]["included"]) & (np.diag(h["shape"]["covariance"]) > 0)
        axis.axhspan(-2, 2, color="0.93")
        axis.axhline(0, color="0.4", lw=0.8)
        axis.plot(centers[supported], np.asarray(h["shape"]["pulls"])[supported], "o", ms=4, color=colors[0])
        if name == "p_reco":
            axis.set_xscale("log")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("(fast − G4) / paired error")
        axis.set_title(_shape_label(h["shape"]), fontsize=10)
    fig.suptitle(f"Cosmic shape pulls — bins ≥{MIN_SHAPE_ENTRIES}/arm; one development parent", fontsize=13)
    fig.supxlabel("Fast: solid-gas branch, no radiative losses, wall off. G4: first-passage steps. "
                  "Both stock reco; not current-default validation.", fontsize=9)
    fig.savefig(Path(destination).with_name("cosmic_pulls.png"), dpi=170)
    plt.close(fig)


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", help="saved paired events.csv.gz")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    result = compare(pd.read_csv(args.events))
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    plot(result, out / "cosmic_trk.png")
    numbers = out / "numbers.json"
    payload = json.loads(numbers.read_text()) if numbers.exists() else {}
    payload["cosmic"] = result
    numbers.write_text(json.dumps(finite_json(payload), indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["efficiency"], indent=2))


if __name__ == "__main__":
    main()
