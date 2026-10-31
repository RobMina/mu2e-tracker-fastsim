"""Single-job tracker tmpfs streaming; no beam dependency or ROOT output handoff.

Uses this core baseline's retained core+tracker source snapshots and input
bindings. SIF is hashed at admission and launch, NOT retained or immutable for
the job lifetime (legacy verify-before-run-not-snapshot policy).
"""
from __future__ import annotations

import json
from dataclasses import asdict
import os
import re
import shlex
import threading
import time
from pathlib import Path
from typing import Any, Dict

from mu2efastsim.endpoints import EndpointError, Resources
from mu2efastsim import site
from mu2efastsim.endpoints.chainpoint import FnalSource
from mu2efastsim.endpoints.artifacts import (binding, checked_record, make_manifest, resolve_image,
                                           resolve_inputs, runtime_prefix, target_identity, verify_cache, verify_result)
from mu2efastsim.endpoints.snapshots import prepare_code
from mu2efastsim.endpoints.submission import submission_lock
from . import kit_task

WAIT_S = 900
RESOURCES = Resources(cores=4, mem_gb=24, gpus=0, gpu_type="", minutes=60)

BODY = r'''module load apptainer >/dev/null 2>&1 || exit 5
M2FS_OUT=$(readlink -f "$M2FS_OUT") || exit 3
CODE=@code@
# Internal inputs and generated ROOT remain in per-job node tmpfs, never shared storage.
L=$(mktemp -d /dev/shm/trkfs_kit_j${SLURM_JOB_ID:-0}_XXXXXXXX) || exit 3
chmod 700 "$L"
trap 'rc=$?; if rm -rf -- "$L" && [ ! -e "$L" ]; then echo "$(date -Is) $(hostname) deleted $L" >> "$M2FS_OUT/cleanup.txt"; else rc=7; fi; exit "$rc"' EXIT
trap 'exit 143' TERM INT
[ "$(stat -f -c %T "$L")" = tmpfs ] || { echo "tracker inputs require tmpfs"; exit 3; }
B() { apptainer exec --cleanenv --env "PYTHONPATH=$CODE/core:$CODE/tracker,PYTHONDONTWRITEBYTECODE=1" -B "$L" -B "$M2FS_OUT" -B "$CODE" --pwd "$CODE" @sif@ "$@"; }
B python3 -c "import trkfastsim.kitpoint" || exit 5
echo "$(hostname) $L" > "$(dirname "$M2FS_OUT")/inbox"
tw=$SECONDS; while [ $((SECONDS-tw)) -lt @waits@ ]; do [ -f "$L/ready" ] && break; sleep 3; done
[ -f "$L/ready" ] || { echo "no input arrived within @waits@ s"; exit 3; }
@input_guard@B python3 -m trkfastsim.kitpoint --gen "$L/gen.root" --gdml "$L/geometry.gdml" \
  --digi-cond "$L/digi_cond.npz" --reco-cond "$L/reco_cond.json" --drift-source "$L/drift_source.root" \
  --seed @seed@ --outdir "$M2FS_OUT" --workdir "$L/work" > "$M2FS_OUT/point.txt" 2>&1 \
  || { tail -20 "$M2FS_OUT/point.txt"; exit 6; }
'''


def validate(params: Dict[str, Any]) -> Dict[str, Any]:
    from mu2efastsim.endpoints.parameters import validate as validate_parameters
    return validate_parameters("tracker_point", params, kit_task.PARAMS, kit_task.DEFAULTS)


class TrackerPointStream:
    def __init__(self, kit, source=None, code=None, background=True, poll_s=10.0):
        self.kit, self.ep = kit, kit.ep
        # Site paths come from the environment (mu2efastsim.site); a missing variable raises SiteConfigError.
        self.source = source or FnalSource(site.fnal_host(),
                                        site.require("FASTSIMKIT_TRACKER_DATA", "M2FS_INPUTS", "M2FS_DATA"))
        self.code = code or site.setting("FASTSIMKIT_TRACKER_CODE") or os.path.join(site.scratch(), "tracker_code_kit")
        self.code_ref = os.environ.get("FASTSIMKIT_CODE_REF", "main")
        self.background, self.poll_s = background, poll_s
        self.dir = Path(kit.out_root).parent / "tracker_point"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def _mark(self, name, what, text=""):
        (self.dir / f"{name}.{what}").write_text(text)

    def _has(self, name, what) -> bool:
        return (self.dir / f"{name}.{what}").exists()

    def _rec(self, name):
        p = self.dir / f"{name}.json"
        return checked_record(p)

    @staticmethod
    def inputs(p) -> Dict[str, str]:
        """Map data-root-relative source paths to fixed tmpfs destinations."""
        return {path: kit_task.DESTINATIONS[key] for key, path in kit_task.input_paths(p).items()}

    def render(self, p, manifest=None):
        from mu2efastsim.endpoints.kit import admit_resources
        from mu2efastsim.endpoints.artifacts import consumption_guard
        p = validate(p)
        admit_resources(self.ep, RESOURCES)
        if not hasattr(self.ep, "stream_in"):
            raise ValueError("fastsimkit: tracker_point requires a streaming endpoint")
        subs = {"sif": shlex.quote(manifest["image"]["path"] if manifest else self.kit.sif),
                "code": shlex.quote(manifest["code"]["path"] if manifest else self.code),
                "waits": str(WAIT_S), "seed": str(p["seed"]),
                "input_guard": consumption_guard('"$L"', manifest["inputs"]["files"]) if manifest else ""}
        body = BODY
        for key, value in subs.items():
            body = body.replace(f"@{key}@", value)
        if manifest is not None:
            body = runtime_prefix(manifest) + body
        return body, RESOURCES

    # -- contract ------------------------------------------------------------------------------------------------
    def check(self, name, params) -> Dict[str, Any]:
        try:
            self.check_binding(name, params)
            self.render(validate(params))
        except ValueError as e:
            return {"ok": False, "message": str(e)}
        return {"ok": True, "message": ""}

    def check_binding(self, name, params, geometry=None):
        if geometry is not None:
            raise ValueError("fastsimkit: tracker_point does not accept geometry files/inputs")
        old = self._rec(name)
        if old is None:
            return None
        p = validate(params)
        target = target_identity(self.ep, self.source)
        if p != old["params"] or target != old["target"]:
            raise ValueError("fastsimkit: different params or target/source from artifact binding")
        return old

    def ensure_code(self):
        return prepare_code(self.ep, self.code, self.code_ref, modules=('core', 'tracker'))

    def submit(self, name, params, geometry=None) -> Dict[str, Any]:
        p = validate(params)
        old = self.check_binding(name, p, geometry)
        if old is None:
            self.render(p)
        target = target_identity(self.ep, self.source)

        def prepare():
            inputs = resolve_inputs(self.source, self.inputs(p))
            code = self.ensure_code()
            image = resolve_image(self.ep, self.kit.sif)
            manifest = make_manifest("tracker_point", code, image, inputs, p, geometry=geometry)
            body, res = self.render(p, manifest)
            return {"params": p, "submitted_at": time.time(), "code": code["path"],
                    "code_ref": self.code_ref, "manifest": manifest,
                    "body": body, "resources": asdict(res), "target": target}

        try:
            rec = binding(self.dir, name, p, target, prepare, geometry=geometry)
            self.ep.submit(name, rec["body"], Resources(**rec["resources"]))
        except EndpointError as e:
            raise ValueError(str(e)) from e
        self.ensure_streaming(name)
        return {"handle": name}

    def ensure_streaming(self, name):
        if "manifest" not in (self._rec(name) or {}):
            return
        if self._has(name, "streamed") or self._has(name, "streamerr"):
            return
        if not self.background:
            self.stream_step(name)
            return
        with self._lock:
            t = self._threads.get(name)
            if t is None or not t.is_alive():
                t = threading.Thread(target=self._loop, args=(name,), daemon=True)
                self._threads[name] = t
                t.start()

    def _loop(self, name):
        while not (self._has(name, "streamed") or self._has(name, "streamerr")):
            if self.stream_step(name) == "gone":
                return
            time.sleep(self.poll_s)

    def stream_step(self, name) -> str:
        """One attempt: 'waiting' (job not started), 'gone' (left the queue), 'streamed', 'error'."""
        st = self.ep.status(name)
        if st.state != "working":
            return "gone"
        try:
            inbox = self.ep.read_file(name, "inbox").split()
            if not inbox:
                return "waiting"
            if (len(inbox) != 2 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", inbox[0])
                    or not re.fullmatch(r"/dev/shm/trkfs_kit_j[0-9]+_[A-Za-z0-9]+", inbox[1])):
                raise ValueError("tracker inbox must identify a per-job tmpfs directory")
            host, tmp = inbox
            rec = self._rec(name)
            if not rec or "manifest" not in rec:
                raise ValueError("legacy streaming record has no artifact manifest")
            entries = rec["manifest"]["inputs"]["files"]
            # Also serializes recovery callers that observed the inbox before the streamed marker.
            with submission_lock(self.dir / ".streams", name):
                if self._has(name, "streamed"):
                    return "streamed"
                if self._has(name, "streamerr"):
                    return "error"
                if self.ep.status(name).state != "working":
                    return "gone"
                have = self.ep.run_on_node(name, host, f"test -f {shlex.quote(tmp)}/ready && echo have || true").strip()
                if have == "have":
                    if not verify_cache(self.ep, name, host, tmp, entries):
                        raise ValueError("ready input cache does not match the bound manifest; refusing overwrite")
                    self._mark(name, "inputsreused", host)
                else:
                    for entry in entries:
                        got = self.ep.stream_in(name, host, f"{tmp}/{entry['dest']}", entry["argv"])
                        if got != entry["md5"]:
                            raise ValueError(f"md5 mismatch for {entry['src']}: {got} != {entry['md5']}")
                    if not verify_cache(self.ep, name, host, tmp, entries):
                        raise ValueError("streamed input cache does not match the bound manifest")
                    self.ep.run_on_node(name, host, f"touch {shlex.quote(tmp)}/ready")
                self._mark(name, "streamed", time.strftime("%Y-%m-%dT%H:%M:%S"))
        except (EndpointError, ValueError, KeyError) as e:
            self._mark(name, "streamerr", str(e))
            self.ep.cancel(name)
            return "error"
        return "streamed"

    def status(self, name) -> Dict[str, Any]:
        from mu2efastsim.endpoints.kit import POLL_MS
        if self._has(name, "streamerr"):
            # Retry cancellation after controller restart / transient cancel failure.
            if self.ep.status(name).state == "working":
                self.ep.cancel(name)
        self.ensure_streaming(name)
        s = self.ep.status(name)
        msg = f"{s.site_state} {s.message}".strip()
        if self._has(name, "streamerr"):
            msg += f"; stream error: {(self.dir / f'{name}.streamerr').read_text()[:200]}"
        return {"state": s.state, "message": msg, "poll_ms": POLL_MS if s.state == "working" else 0, "progress": None}

    def results(self, name) -> Dict[str, Any]:
        s = self.ep.status(name)
        if s.state != "completed":
            raise ValueError(f"fastsimkit: {name!r} is {s.state}, not completed")
        files = self.ep.fetch(name, self.kit.out_root)
        out = Path(self.kit.out_root) / name / "out"
        if not (out / "cleanup.txt").exists():
            raise ValueError(f"fastsimkit: {name!r} staged Mu2e-internal inputs but wrote no cleanup record")
        mpath = out / "metrics.json"
        if not mpath.exists():
            raise ValueError(f"fastsimkit: {name!r} wrote no out/metrics.json")
        rec = self._rec(name)
        if rec is None:
            raise ValueError(f"fastsimkit: {name!r} has no task record")
        verify_result(out, rec["manifest"])
        if self._has(name, "streamerr"):
            raise ValueError("fastsimkit: tracker_point input streaming failed")
        if not (out / "provenance.json").is_file():
            raise ValueError("fastsimkit: tracker_point wrote no provenance.json")
        metrics = kit_task.validate_metrics(json.loads(mpath.read_text()))
        return {"metrics": metrics,
                "files": [{"name": p.name, "uri": p.resolve().as_uri(), "kind": "text"} for p in files
                          if p.suffix in (".json", ".txt", ".out", ".log")],
                "metadata": {"endpoint": self.ep.name, "job_id": s.job_id, "site_state": s.site_state,
                             "artifact_manifest": rec["manifest"], "manifest_sha": rec["manifest"]["manifest_sha"],
                             "artifact_binding": "verified",
                             "code_ref": rec["code_ref"]}}

    def cancel(self, name) -> Dict[str, Any]:
        return {"state": self.ep.cancel(name)}
