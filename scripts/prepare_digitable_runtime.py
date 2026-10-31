"""Make task-private sequence runners from the umbrella's already-validated Offline runner.

No builds, installs or transfers. Repositories and the saved runtime client remain unchanged.
"""
import argparse
from pathlib import Path


REPOSITORIES = "config-osg.opensciencegrid.org mu2e.opensciencegrid.org fermilab.opensciencegrid.org"
CALL = '"$OFFLINE_TIME" -v -o "${OFFLINE_LOG_PREFIX}.art.time" mu2e "${args[@]}"'


def prepare(source, output):
    source, output = Path(source), Path(output)
    text = source.read_text()
    if text.count(REPOSITORIES) != 2 or text.count(CALL) != 1:
        raise ValueError("unsupported umbrella runtime version; inspect before adapting")
    text = text.replace(REPOSITORIES, REPOSITORIES.replace(" fermilab", " mu2e-development.opensciencegrid.org fermilab"))
    text = text.replace("OFFLINE_OUTPUT_FILE OFFLINE_EVENTS OFFLINE_FCL_PATH OFFLINE_WARM_REPEAT;",
                        "OFFLINE_OUTPUT_FILE OFFLINE_EVENTS OFFLINE_FCL_PATH OFFLINE_WARM_REPEAT PAYLOAD;")
    output.mkdir(parents=True, exist_ok=True)
    for name, script in (("offline_sequence.sh", "run_digitable_sequence.sh"), ("offline_fast.sh", "run_fastdigis_reco.sh"),
                         ("offline_payload_diagnostic.sh", "run_payload_diagnostic.sh"),
                         ("offline_payload_control.sh", "run_payload_control.sh")):
        runner = text.replace(CALL, f'/bin/bash "$AREA/{script}" "$FCL" "$@"')
        with (output / name).open("x") as stream:
            stream.write(runner)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    prepare(args.runner, args.output)


if __name__ == "__main__":
    main()
