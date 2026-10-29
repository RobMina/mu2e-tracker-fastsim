"""Static/pure source checks only: no native compile, fit or geometry evaluation."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "crossing_geometry_patch", ROOT / "art/crossing_capture/patch_geometry.py")
PATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCH)
HEADER = (ROOT / "art/crossing_capture/CrossingGeometryWriter.hh").read_text()


def source():
    return (PATCH._INCLUDE + PATCH._MEMBER + PATCH._BEGIN +
            '        if (fullSchedule_) {\n' + PATCH._GUARD + PATCH._CALL +
            '          StockTrack::IterationCapture capture(iterationCaptureRequested_ ? &iterations : nullptr);\n' +
            PATCH._BUILD + '        }\n')


class GeometryPatchTest(unittest.TestCase):
    def test_additive_only(self):
        original = source()
        result = PATCH.patch(original)
        for extra in (
            '#include "TrkFastSim/inc/CrossingGeometryWriter.hh"\n',
            '    CrossingGeometryWriter crossingGeometry_;\n',
            '    if (fullSchedule_) crossingGeometry_.beginJob(*tfs);\n',
            PATCH._INSERT,
        ):
            self.assertEqual(result.count(extra), 1)
            result = result.replace(extra, '', 1)
        self.assertEqual(result, original)

    def test_selected_only_before_native_capture(self):
        result = PATCH.patch(source())
        self.assertIn('if (iterationCaptureRequested_) {', PATCH._INSERT)
        self.assertIn('try {', PATCH._INSERT)
        self.assertIn('catch (...)', PATCH._INSERT)
        self.assertLess(result.index('crossingGeometry_.write'), result.index('StockTrack::IterationCapture'))
        self.assertNotIn('iterationCaptureErrors_', PATCH._INSERT)
        self.assertNotIn('iterationCaptureValid_', PATCH._INSERT)
        self.assertNotIn('->Branch', result)

    def test_anchor_fail_closed(self):
        for anchor in (PATCH._INCLUDE, PATCH._MEMBER, PATCH._BEGIN, PATCH._CALL, PATCH._GUARD):
            with self.subTest(anchor=anchor):
                for changed in (source().replace(anchor, ''), source() + anchor,
                                source().replace(anchor, anchor.replace(' ', '\t', 1))):
                    if changed == source():
                        continue
                    with self.assertRaises(ValueError):
                        PATCH.patch(changed)

    def test_order_and_capture_scope_required(self):
        for changed in (
            source().replace(PATCH._BUILD, ''),
            source().replace('StockTrack::IterationCapture capture(', 'unrecognized('),
            source().replace(PATCH._GUARD, '').replace(PATCH._BUILD, PATCH._GUARD + PATCH._BUILD),
        ):
            with self.assertRaises(ValueError):
                PATCH.patch(changed)

    def test_repeat_rejected(self):
        with self.assertRaises(ValueError):
            PATCH.patch(PATCH.patch(source()))

    def test_raw_transform_native_endpoints_and_float(self):
        for accessor in ('xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz'):
            self.assertIn('r.' + accessor + '()', HEADER)
        self.assertIn('transform(panel->dsToPanel())', HEADER)
        self.assertIn('transform.displacement()', HEADER)
        self.assertIn('float halfLength_', HEADER)
        for operand in ('strawEnd(StrawEnd::cal)', 'strawEnd(StrawEnd::hv)',
                        'wireEnd(StrawEnd::cal)', 'wireEnd(StrawEnd::hv)',
                        'getMidPoint()', 'wireDirection()', 'planeExists(plane.id())'):
            self.assertIn(operand, HEADER)
        for operation in ('findTCA(', 'updateReference(', 'updateState(', '.Unit(', '->Unit('):
            self.assertNotIn(operation, HEADER)

    def test_status_and_root_failure_gate(self):
        self.assertIn('tree->Fill() < 0', HEADER)
        self.assertIn('status_ = 0; error_.clear()', HEADER)
        self.assertIn('status_ = 1; fill(bindings_)', HEADER)
        self.assertIn('status_ = -1', HEADER)
        self.assertIn('std::rethrow_exception(failure)', HEADER)
        self.assertIn('panelCount == npanels_ && strawCount == nstraws_', HEADER)


if __name__ == '__main__':
    unittest.main()
