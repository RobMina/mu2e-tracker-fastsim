"""Summarize an isolated pytest-file log without hiding a failed or unfinished file."""
import argparse
import json
from pathlib import Path
import re

from scripts.transport_capture_audit import digest

STATUS = re.compile(r"(\d+) (passed|failed|skipped|xfailed|xpassed|errors?|subtests passed)")


def summarize(text, expected):
    rows, current = [], None
    for line in text.splitlines():
        if line.startswith("==== tests/"):
            current = dict(file=line[5:].strip(), counts={}, complete=False)
            rows.append(current)
        elif current is not None and " in " in line and ("passed" in line or "failed" in line or "error" in line or "skipped" in line):
            counts = {name: int(number) for number, name in STATUS.findall(line)}
            if counts:
                current["counts"] = counts
                current["complete"] = True
                current["summary"] = line.strip("= ")
    names = [row["file"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate pytest file transactions")
    missing = sorted(set(expected) - set(names))
    unexpected = sorted(set(names) - set(expected))
    unfinished = [row["file"] for row in rows if not row["complete"]]
    totals = {name: sum(row["counts"].get(name, 0) for row in rows)
              for name in ("passed", "failed", "skipped", "xfailed", "xpassed", "error", "errors", "subtests passed")}
    return dict(files_started=len(rows), files_expected=len(expected), missing=missing, unexpected=unexpected,
        unfinished=unfinished, totals=totals, files=rows,
        complete=not (missing or unexpected or unfinished),
        no_unexpected_failures=not any(totals[name] for name in ("failed", "xpassed", "error", "errors")))


def run(log, selection, out):
    log, selection, out = map(Path, (log, selection, out))
    hashes = {str(p): digest(p) for p in (log, selection)}
    expected = [line.strip() for line in selection.read_text().splitlines() if line.startswith("tests/")]
    result = summarize(log.read_text(), expected)
    result.update(input_sha256=hashes, scope="isolated pytest log transactions; scheduler exit and source hashes are separate gates")
    if hashes != {str(p): digest(p) for p in (log, selection)}:
        raise ValueError("test log/selection changed during summary")
    with out.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--selection", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.log, args.selection, args.out)
