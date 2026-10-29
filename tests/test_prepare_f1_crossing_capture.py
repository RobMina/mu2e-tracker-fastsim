"""Synthetic preparation/source guards; not a native compile or passivity claim.

No real git, external checkout, compiler, build, or physics execution is permitted.
Anchor fixtures intentionally do not depend on a locally stale native checkout.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_f1_crossing_capture as P


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes()
            for p in root.rglob("*") if p.is_file() and not p.is_symlink()}


def strict_report(path):
    def invalid_constant(value):
        pytest.fail("non-finite JSON constant: " + value)

    def unique_keys(items):
        result = {}
        for key, value in items:
            assert key not in result, "duplicate JSON key: " + key
            result[key] = value
        return result

    return json.loads(path.read_text(), parse_constant=invalid_constant,
                      object_pairs_hook=unique_keys)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    base, area = tmp_path / "operand-base", tmp_path / "crossing-area"
    templates = tmp_path / "templates"
    # Use the actual templates, but only in isolated local scratch space.
    shutil.copytree(P.TEMPLATES, templates,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    frozen_names = tuple(P.FROZEN)
    for name in frozen_names:
        put(base / name, "// inherited operand source: " + name + "\n")
    put(base / "TrkFastSim/fcl/synthetic.fcl", "capture: false\n")
    put(base / "TrkFastSim/data/payload.txt", "local package payload\n")
    pins = {name: str(i) * 40 for i, name in enumerate(P.PINS, 1)}
    for repo in pins:
        put(base / repo / "tracked.txt", "clean tracked content: " + repo + "\n")
    frozen = {name: sha(base / name) for name in frozen_names}
    clean = snapshot(base)
    # Worktree creation starts from the pinned commit, not the dirty operand tree.
    clean["KinKal/Fit/Track.hh"] = b"// clean pinned track, no operand observer\n"
    del clean["KinKal/Fit/IterationTrace.hh"]
    calls = []
    statuses = {repo: ("M Fit/Track.hh\n?? Fit/IterationTrace.hh" if repo == "KinKal" else "")
                for repo in pins}
    heads = dict(pins)

    def fake_git(repo, *args):
        repo = Path(repo)
        calls.append((repo, args))
        if repo.parent == base and repo.name in pins:
            if args == ("rev-parse", "HEAD"):
                return heads[repo.name]
            if args == ("status", "--porcelain", "--untracked-files=all"):
                return statuses[repo.name]
            if args == ("worktree", "add", "--detach", str(area / repo.name), pins[repo.name]):
                for name, data in clean.items():
                    if Path(name).parts[0] == repo.name:
                        dest = area / name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(data)
                return ""
        if repo.parent == area and repo.name in pins and args == ("diff", "--check"):
            return ""
        pytest.fail("unexpected git or native action: " + repr((repo, args)))

    def fake_patched(sources):
        result = dict(sources)
        result["KinKal/Fit/Track.hh"] += "// synthetic crossing observer\n"
        result["Offline/Mu2eKinKal/inc/KKStrawXing.hh"] += "// synthetic crossing hook\n"
        return result

    for key, value in (("BASE", base), ("AREA", area), ("TEMPLATES", templates),
                       ("FROZEN", frozen), ("PINS", pins), ("git", fake_git),
                       ("patched", fake_patched)):
        monkeypatch.setattr(P, key, value)

    def forbidden(*args, **kwargs):
        pytest.fail("subprocess/native execution is forbidden in synthetic tests")

    monkeypatch.setattr(P.subprocess, "Popen", forbidden)
    return SimpleNamespace(base=base, area=area, report=tmp_path / "report.json",
                           templates=templates, frozen=frozen, pins=pins, calls=calls,
                           statuses=statuses, heads=heads, before=snapshot(base),
                           transform=fake_patched, fake_git=fake_git)


def run(s, **kwargs):
    result = P.prepare(s.base, s.area, s.report, **kwargs)
    assert strict_report(s.report) == result
    assert result["preparer_sha256"] == sha(Path(P.__file__))
    assert result["pins"] == s.pins
    assert result["frozen_sources"] == s.frozen
    for flag in ("build_run", "physics_run", "runtime_passivity_validated"):
        assert result[flag] is False
    assert set(result["before_sha256"]) == set(result["end_sha256"])
    for path, actual in result["end_sha256"].items():
        assert actual == (sha(Path(path)) if Path(path).is_file() else None)
    assert result["hashes_unchanged"] == bool(result["before_sha256"] and
                                               result["before_sha256"] == result["end_sha256"])
    return result


def assert_rejected(s, result, message):
    assert result["completed"] is False
    assert message in result["error"]
    assert not s.area.exists()
    assert not any(args[:2] == ("worktree", "add") for _, args in s.calls)


def test_prepare_installs_templates_inherits_only_allowed_dirty_sources(sandbox):
    s = sandbox
    result = run(s)
    assert result["completed"] is True
    assert result["validated_only"] is False
    assert result["hashes_unchanged"] is True
    assert snapshot(s.base) == s.before
    inherited = (s.base / "KinKal/Fit/Track.hh").read_text()
    assert (s.area / "KinKal/Fit/Track.hh").read_text() == inherited + "// synthetic crossing observer\n"
    assert (s.area / "KinKal/Fit/IterationTrace.hh").read_bytes() == (s.base / "KinKal/Fit/IterationTrace.hh").read_bytes()
    for template, destination in P.INSTALL.items():
        assert (s.area / destination).read_bytes() == (s.templates / template).read_bytes()
        assert result["templates"][str(s.templates / template)] == sha(s.templates / template)
    for name, digest in result["package_sources"].items():
        assert name.startswith("TrkFastSim/")
        assert sha(s.base / name) == digest
        assert sha(s.area / name) == digest  # synthetic transform does not change this package
    expected_paths = set(s.frozen) | set(result["package_sources"]) | set(P.INSTALL.values())
    assert set(result["prepared_sources"]) == expected_paths
    assert result["prepared_sources"] == {name: sha(s.area / name) for name in expected_paths}
    manifest = dict(line.split("  ", 1)[::-1] for line in
                    (s.area / "capture-source-manifest.sha256").read_text().splitlines())
    assert manifest == result["prepared_sources"]
    assert (s.area / "source-pins.txt").read_text() == "".join(f"{repo} {pin}\n" for repo, pin in s.pins.items())
    assert {repo.name for repo, args in s.calls if args[:2] == ("worktree", "add")} == set(s.pins)
    assert {repo.name for repo, args in s.calls if args == ("diff", "--check")} == set(s.pins)


def test_validate_only_transforms_and_watches_without_creating_area(sandbox, monkeypatch):
    s = sandbox
    seen = []

    def transform(sources):
        seen.append(dict(sources))
        return s.transform(sources)

    monkeypatch.setattr(P, "patched", transform)
    result = run(s, validate_only=True)
    assert result["completed"] and result["validated_only"] and result["hashes_unchanged"]
    assert len(seen) == 1
    assert not s.area.exists()
    assert "prepared_sources" not in result
    assert result["changed_paths"] == ["KinKal/Fit/Track.hh", "Offline/Mu2eKinKal/inc/KKStrawXing.hh"]
    assert all(args[0] in ("rev-parse", "status") for _, args in s.calls)
    assert snapshot(s.base) == s.before


@pytest.mark.parametrize("which", ["base", "area", "same"])
def test_path_pinning_rejects_before_git(sandbox, which):
    s = sandbox
    base = s.base.parent / "wrong-base" if which == "base" else s.base
    area = s.base if which == "same" else s.area.parent / "wrong-area" if which == "area" else s.area
    result = P.prepare(base, area, s.report)
    assert strict_report(s.report) == result
    assert not result["completed"] and "only the named" in result["error"]
    assert not result["hashes_unchanged"]
    assert not s.calls
    assert snapshot(s.base) == s.before


@pytest.mark.parametrize("existing", ["directory", "file", "dangling-symlink"])
def test_destination_never_overwritten(sandbox, existing):
    s = sandbox
    if existing == "directory":
        put(s.area / "keep", "do not overwrite\n")
    elif existing == "file":
        s.area.write_text("do not overwrite\n")
    else:
        s.area.symlink_to(s.area.parent / "nonexistent")
    result = run(s)
    assert not result["completed"] and "existing" in result["error"]
    assert not s.calls
    if existing == "directory":
        assert (s.area / "keep").read_text() == "do not overwrite\n"
    elif existing == "file":
        assert s.area.read_text() == "do not overwrite\n"
    else:
        assert s.area.is_symlink()
    assert snapshot(s.base) == s.before


@pytest.mark.parametrize("existing", ["file", "symlink"])
def test_report_is_exclusive_even_before_validation(sandbox, existing):
    s = sandbox
    protected = s.report if existing == "file" else s.report.parent / "protected.json"
    protected.write_bytes(b"immutable prior report\n")
    if existing == "symlink":
        s.report.symlink_to(protected)
    with pytest.raises(FileExistsError):
        P.prepare(s.base, s.area, s.report)
    assert protected.read_bytes() == b"immutable prior report\n"
    assert not s.calls and not s.area.exists()
    assert snapshot(s.base) == s.before


@pytest.mark.parametrize("failure", ["hash", "missing", "pin", "offline-dirty", "kinkal-extra", "kinkal-clean"])
def test_frozen_hash_pin_and_status_fail_closed(sandbox, failure):
    s = sandbox
    source = s.base / "KinKal/Fit/Track.hh"
    if failure == "hash":
        source.write_text("unexpected source\n")
    elif failure == "missing":
        source.unlink()
    elif failure == "pin":
        s.heads["Offline"] = "0" * 40
    elif failure == "offline-dirty":
        put(s.base / "Offline/unreviewed.txt", "must not be copied\n")
        s.statuses["Offline"] = "?? unreviewed.txt"
    elif failure == "kinkal-extra":
        put(s.base / "KinKal/unreviewed.txt", "must not be copied\n")
        s.statuses["KinKal"] += "\n?? unreviewed.txt"
    else:
        s.statuses["KinKal"] = ""
    before = snapshot(s.base)
    result = run(s)
    expected = {"hash": "hash differs", "missing": "FileNotFoundError", "pin": "pin differs"}.get(failure, "worktree changes")
    assert_rejected(s, result, expected)
    assert snapshot(s.base) == before
    if failure in ("hash", "missing"):
        assert not s.calls


@pytest.mark.parametrize("kind", ["file", "directory", "dangling"])
def test_package_symlinks_are_rejected_not_followed_or_copied(sandbox, kind):
    s = sandbox
    outside = s.base.parent / "outside"
    if kind == "directory":
        put(outside / "secret", "not package data\n")
    elif kind == "file":
        outside.write_text("not package data\n")
    (s.base / "TrkFastSim/linked").symlink_to(outside, target_is_directory=kind == "directory")
    result = run(s)
    assert_rejected(s, result, "symlink")
    assert snapshot(s.base) == s.before


@pytest.mark.parametrize("mutation", ["none", "source", "template", "remove-package"])
def test_transform_failure_keeps_known_provenance_and_honest_end_hashes(sandbox, monkeypatch, mutation):
    s = sandbox

    def fail(sources):
        if mutation == "source":
            (s.base / "KinKal/Fit/Track.hh").write_text("changed during preparation\n")
        elif mutation == "template":
            (s.templates / "CrossingTrace.hh").write_text("changed template\n")
        elif mutation == "remove-package":
            (s.base / "TrkFastSim/data/payload.txt").unlink()
        raise ValueError('synthetic anchor failure with "quotes"\nand newline')

    monkeypatch.setattr(P, "patched", fail)
    result = run(s)
    assert_rejected(s, result, "synthetic anchor failure")
    watched = result["before_sha256"]
    for name, digest in s.frozen.items():
        assert watched[str(s.base / name)] == digest
    for name in (*P.INSTALL, *P.PATCH_FILES):
        assert str(s.templates / name) in watched
    assert str(s.base / "TrkFastSim/data/payload.txt") in watched
    assert result["hashes_unchanged"] is (mutation == "none")
    if mutation == "none":
        assert snapshot(s.base) == s.before


def test_input_change_even_after_successful_transform_blocks_completion(sandbox, monkeypatch):
    s = sandbox

    def mutate(sources):
        result = s.transform(sources)
        (s.base / "KinKal/Fit/Track.hh").write_text("changed during preparation\n")
        return result

    monkeypatch.setattr(P, "patched", mutate)
    result = run(s, validate_only=True)
    assert not result["completed"] and not result["hashes_unchanged"]
    assert not s.area.exists()


@pytest.mark.parametrize("phase", ["worktree", "diff"])
def test_checkout_or_diff_failure_is_recorded_without_mutating_base(sandbox, monkeypatch, phase):
    s = sandbox

    def fail_git(repo, *args):
        if args[0] == phase:
            raise OSError("synthetic " + phase + " failure")
        return s.fake_git(repo, *args)

    monkeypatch.setattr(P, "git", fail_git)
    result = run(s)
    assert not result["completed"] and result["hashes_unchanged"]
    assert "synthetic " + phase + " failure" in result["error"]
    assert snapshot(s.base) == s.before
    if phase == "diff":
        assert result["prepared_sources"] == {name: sha(s.area / name) for name in result["prepared_sources"]}


@pytest.fixture
def core():
    spec = importlib.util.spec_from_file_location("crossing_core_test", ROOT / "art/crossing_capture/patch_core.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def anchor_source(core, function):
    """Extract only literal input anchors, never a whole local native source file."""
    tree = ast.parse(Path(core.__file__).read_text())
    body = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
    anchors = [ast.literal_eval(node.args[1]) for node in ast.walk(body)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "replace"]
    if function == "xing":
        anchors += ["    varscale_(1.0)\n  {}"] * 2
    return "\n// independent fixture boundary\n".join(anchors) + "\n// untouched physics sentinel\n"


@pytest.mark.parametrize("function", ["track", "xing"])
def test_core_anchor_transform_additive_and_second_application_refused(core, function):
    original = anchor_source(core, function)
    transform = getattr(core, function)
    changed = transform(original)
    assert changed != original
    assert changed.endswith("// untouched physics sentinel\n")
    with pytest.raises(ValueError):
        transform(changed)
    if function == "track":
        assert "crossings_(sink ? crossings : nullptr)" in changed
        assert "crossingObserver_(sink ? crossingObserver : nullptr)" in changed
        assert changed.index("out.change = crossing->params();") < changed.index("iterationCapture_->crossingObserver_(*crossing")
        assert "for(auto& effptr : effects_) effptr->updateReference(*ptraj,miconfig);" in changed
        assert "effects_.emplace_back(std::make_unique<KKMAT>(exing,*fittraj_,miconfig));" in changed
    else:
        assert changed.count("{ captureCrossing(0); }") == 2
        assert "if(!sink)return;" in changed
        assert "return radius/speed;" in changed
        assert "crossingSink ? &crossingSearch : nullptr" in changed
        assert changed.index("ca_ = static_cast<CA>(pca);") < changed.index("if(crossingSink)captureCrossing")
        assert "void setHit(KKSTRAWHITPTR& shptr) { shptr_ = shptr; captureCrossing(4); }" in changed


@pytest.mark.parametrize("function", ["track", "xing"])
@pytest.mark.parametrize("fault", ["missing", "duplicate"])
def test_core_rejects_ambiguous_or_missing_source_anchor(core, function, fault):
    text = anchor_source(core, function)
    anchor = ('#include "KinKal/Fit/IterationTrace.hh"' if function == "track"
              else '#include "Offline/Mu2eKinKal/inc/KKStrawHit.hh"')
    text = text.replace(anchor, "" if fault == "missing" else anchor + "\n" + anchor)
    with pytest.raises(ValueError, match="source anchor is not unique"):
        getattr(core, function)(text)


@pytest.mark.parametrize("constructors", [0, 1, 3])
def test_xing_requires_exactly_two_constructor_anchors(core, constructors):
    text = anchor_source(core, "xing")
    before = "    varscale_(1.0)\n  {}"
    text = text.replace(before, "") + ("\n" + before) * constructors
    with pytest.raises(ValueError, match="expected both crossing constructors"):
        core.xing(text)


def test_report_inside_package_cannot_claim_success_with_stale_end_hashes(sandbox):
    """A report must not silently become a mutable source package input.

    Refusal before opening is acceptable, as is a failed report with truthful
    final hashes; successful self-hashing before writing the JSON is not.
    """
    s = sandbox
    s.report = s.base / "TrkFastSim/self-report.json"
    try:
        result = P.prepare(s.base, s.area, s.report, validate_only=True)
    except ValueError:
        assert snapshot(s.base) == s.before
        assert not s.area.exists()
        return
    assert strict_report(s.report) == result
    assert result["completed"] is False, "report creation must not mutate the validated operand base and claim success"
    for path, digest in result["end_sha256"].items():
        assert digest == (sha(Path(path)) if Path(path).is_file() else None)
    assert not s.area.exists()


def test_preparer_has_no_native_build_or_runtime_command():
    text = Path(P.__file__).read_text()
    for forbidden in ("muse setup", "muse build", "cmake --build", "scons", "mu2e -c", "build_kinkal_into_area"):
        assert forbidden not in text
