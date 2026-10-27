"""tracker_point: fixed primary-only stage C -> digi -> hybrid F2 -> writer.

No secondary library, no F1/stock-event fitting. F2 uses 2000 MeV/c and no
material. Endpoint outputs are JSON metadata only; EventNtuple stays in scratch.
Paths below are relative to the DATA ROOT, not its crv_fastsim child.
"""
from __future__ import annotations

PARAMS = ("sample", "shard", "seed")
SAMPLE = "extracted_v04_primary_v1"
DEFAULTS = {"sample": SAMPLE, "shard": 11, "seed": 1}
METRICS = ("generated_events", "primary_muons", "events_with_steps", "straw_steps",
           "digis", "hits", "seeds", "fit_attempts", "fit_successes", "events_with_fit",
           "written_events", "fit_event_fraction", "fit_event_fraction_se", "active_hits",
           "steps_s", "digitization_s", "reco_s", "writer_s", "wall_s")
INPUT_PATHS = {
    "gen": "crv_fastsim/g4ref_v1_tables/gen.001400_{shard:08d}.root",
    "gdml": "crv_fastsim/geometry/mu2e_extracted_v04.gdml",
    "digi_cond": "trkfs_digi_20261026/v2/digicond.npz",
    "reco_cond": "trk_fastreco/cond_mc_au.json",
    # Conditions tree ONLY: never the stock events/hits/seed/fit populations.
    "drift_source": "trk_fastreco/trk_sa2/tracks_ref_mc_saveall.001400_00000000.root",
}
DESTINATIONS = {"gen": "gen.root", "gdml": "geometry.gdml", "digi_cond": "digi_cond.npz",
                "reco_cond": "reco_cond.json", "drift_source": "drift_source.root"}
LABELS = {"scenario": SAMPLE, "transport": "stage C sliced MSC; all primary muons",
          "secondary_library": False, "fit": "hybrid F2", "momentum_MeV_c": 2000,
          "fit_material": False, "stock_event_conditioning": False,
          "denominator": "all generated events, including events without steps or fits",
          "retention": "JSON metadata only; EventNtuple writer checked in job scratch"}


def input_paths(params):
    from mu2efastsim.endpoints.parameters import validate
    p = validate("tracker_point", params, PARAMS, DEFAULTS)
    return {key: path.format(shard=p["shard"]) for key, path in INPUT_PATHS.items()}


def validate_metrics(metrics):
    """Require the whole finite pipeline contract, even without requested_metrics."""
    import math
    if not isinstance(metrics, dict) or set(metrics) != set(METRICS):
        raise ValueError("fastsimkit: tracker_point requires the complete pipeline metrics")
    counts = set(METRICS) - {"fit_event_fraction", "fit_event_fraction_se", "steps_s", "digitization_s",
                             "reco_s", "writer_s", "wall_s"}
    for name, value in metrics.items():
        try:
            valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"fastsimkit: tracker_point metric {name} must be finite and nonnegative")
        if name in counts and value != int(value):
            raise ValueError(f"fastsimkit: tracker_point metric {name} must be a whole count")
    if metrics["written_events"] != metrics["generated_events"]:
        raise ValueError("fastsimkit: tracker_point writer must preserve all generated events")
    n, k = metrics["generated_events"], metrics["events_with_fit"]
    if n <= 0 or k > n or metrics["events_with_steps"] > n:
        raise ValueError("fastsimkit: tracker_point requires a nonempty full generated denominator")
    if (metrics["fit_successes"] > metrics["fit_attempts"] or
            metrics["fit_attempts"] != metrics["seeds"] or k > metrics["fit_successes"] or
            metrics["hits"] != metrics["digis"]):
        raise ValueError("fastsimkit: tracker_point inconsistent hit/fit waterfall")
    fraction = k / n
    se = math.sqrt(fraction * (1 - fraction) / n)
    if not (math.isclose(metrics["fit_event_fraction"], fraction, abs_tol=1e-12) and
            math.isclose(metrics["fit_event_fraction_se"], se, abs_tol=1e-12)):
        raise ValueError("fastsimkit: tracker_point fraction/SE must use the full generated denominator")
    return metrics


# EAF's existing fnal_data configuration names the crv_fastsim child, not DATA.
BODY = r'''@runtime_a@D=@fnal_data@/..
L=$(mktemp -d "${TMPDIR:-/tmp}/trkfs_kit_XXXXXXXX") || exit 3
chmod 700 "$L"
trap 'rc=$?; if rm -rf -- "$L" && [ ! -e "$L" ]; then echo "$(date -Is) $(hostname) deleted $L" >> "$M2FS_OUT/cleanup.txt"; else rc=7; fi; exit "$rc"' EXIT
trap 'exit 143' TERM INT
SHARD=$(printf '%08d' @p_shard@)
GEN="$D/crv_fastsim/g4ref_v1_tables/gen.001400_${SHARD}.root"
GDML="$D/crv_fastsim/geometry/mu2e_extracted_v04.gdml"
DIGI="$D/trkfs_digi_20261026/v2/digicond.npz"
RECO="$D/trk_fastreco/cond_mc_au.json"
DRIFT="$D/trk_fastreco/trk_sa2/tracks_ref_mc_saveall.001400_00000000.root"
for f in "$GEN" "$GDML" "$DIGI" "$RECO" "$DRIFT"; do [ -r "$f" ] || { echo "missing $f"; exit 3; }; done
A python -m trkfastsim.kitpoint --gen "$GEN" --gdml "$GDML" --digi-cond "$DIGI" \
  --reco-cond "$RECO" --drift-source "$DRIFT" --seed @p_seed@ --outdir "$M2FS_OUT" --workdir "$L" \
  > "$M2FS_OUT/point.txt" 2>&1 || { tail -20 "$M2FS_OUT/point.txt"; exit 6; }
'''


def register(tasks, task_cls, resources_cls, fnal_tasks):
    tasks["tracker_point"] = task_cls(params=PARAMS, metrics=METRICS, defaults=DEFAULTS,
        resources=resources_cls(cores=4, mem_gb=24, gpus=0, gpu_type="", minutes=60), body=BODY)
    fnal_tasks.add("tracker_point")


class _UVADelegate:
    @staticmethod
    def applies(endpoint):
        return not getattr(endpoint, "fnal_data", None) and hasattr(endpoint, "stream_in")

    @staticmethod
    def make(kit):
        from .kit_uva import TrackerPointStream
        return TrackerPointStream(kit)


def register_delegate(delegates):
    delegates["tracker_point"] = _UVADelegate
