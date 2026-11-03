"""steps CLI defaults: stage C with sliced MSC (the production invocation); stage A and one-|p| MSC stay explicit."""
from trkfastsim import cli


def _steps(*extra):
    return cli.build_parser().parse_args(["steps", "gen.root", *extra])


def test_defaults_are_stage_c_with_sliced_msc():
    a = _steps()
    assert (a.stage, a.sliced_msc) == ("C", True)


def test_stage_a_and_unsliced_msc_remain_available():
    a = _steps("--stage", "A", "--no-sliced-msc")
    assert (a.stage, a.sliced_msc) == ("A", False)
    assert _steps("--stage", "C", "--sliced-msc").sliced_msc
