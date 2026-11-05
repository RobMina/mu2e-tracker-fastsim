"""Portable numeric seed guard from the actual cosmic stock-chain wrapper."""
from pathlib import Path
import subprocess

import pytest

SCRIPT=Path(__file__).parents[1]/'scripts/cosmic_wall_reco_uva.sh'


@pytest.mark.parametrize('seeds,success', [('20261102',True),('2.0261102e7',True),('20261103',False),
                                          ('2.0261103e7',False),('',False),('20261102\n20261102',False)])
def test_expanded_seed_guard_accepts_equivalent_numeric_serialization_only(tmp_path,seeds,success):
    line=next(x for x in SCRIPT.read_text().splitlines() if x.startswith("awk '/^[[:space:]]*baseSeed:/"))
    guard=line.replace('\\$','$')
    (tmp_path/'fcl').mkdir()
    (tmp_path/'fcl/digi.expanded.fcl').write_text('\n'.join('baseSeed: '+x for x in seeds.splitlines()))
    result=subprocess.run(['bash','-c',guard],cwd=tmp_path,capture_output=True,text=True)
    assert (result.returncode==0)==success


def test_wrappers_do_not_launch_build_or_g4_or_delete_closed_products():
    root=SCRIPT.parent
    for name in ('cosmic_wall_pilot_uva.sh','cosmic_wall_reco_uva.sh','cosmic_wall_compare_uva.sh'):
        text=(root/name).read_text()
        for forbidden in ('muse build','muse tarball','scons ','cmake --build','rm -f inj.art','rm -rf','jobsub_submit'):
            assert forbidden not in text
        subprocess.run(['bash','-n',str(root/name)],check=True)
