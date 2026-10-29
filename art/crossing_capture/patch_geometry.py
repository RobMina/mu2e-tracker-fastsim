"""Pure fail-closed patch of the frozen full-schedule TrkFastSim module.

No I/O, builds or physics calls. The caller must copy CrossingGeometryWriter.hh
into TrkFastSim/inc and verify frozen Offline APIs separately. Existing module
schemas and capture validity/error fields are untouched. Geometry errors live
in crossing_geometry_bindings, and must be checked by an independent gate:
selected targets require attempted + complete status, exact counts/full keys;
missing, failed or duplicate terminal bindings fail that gate. The writer records
errors and rethrows; the module catches outside the native fit to preserve physics.
CaptureIterations remains the existing default-off selection switch.
"""

_INCLUDE = '#include "TrkFastSim/inc/IterationCaptureWriter.hh"\n'
_MEMBER = '    IterationCaptureWriter<KTRAJ> iterationWriter_;\n'
_BEGIN = '    if (fullSchedule_) iterationWriter_.beginJob(*tfs);\n'
_CALL = '          iterationWriter_.setIdentity(run_, subrun_, event_, iseed);\n'
_GUARD = '          iterationCaptureRequested_ = captureSelected(iseed);\n'
_BUILD = '          auto trk = build(config_);\n'
_INSERT = '''          if (iterationCaptureRequested_) {
            // Independent diagnostics BEFORE native construction; never change fit/capture status.
            // Writer records attempted/terminal binding; missing terminal is a gate failure.
            try {
              crossingGeometry_.write(*tracker, run_, subrun_, event_, iseed);
            } catch (...) {
              // Geometry failure is isolated from physics. Check the standalone bindings.
            }
          }
'''


def patch(moduletext):
    """Return additive patched text; reject drift, ambiguity and repeated patching."""
    if 'CrossingGeometryWriter' in moduletext or 'crossingGeometry_' in moduletext:
        raise ValueError('crossing geometry is already patched')
    for anchor in (_INCLUDE, _MEMBER, _BEGIN, _CALL, _GUARD):
        if moduletext.count(anchor) != 1:
            raise ValueError('missing/ambiguous geometry patch anchor: ' + anchor.strip())
    # The full-schedule build must follow the selected-target assignment and call;
    # other (stage-by-stage) builds exist, and must NOT receive geometry capture.
    guard = moduletext.index(_GUARD)
    call = moduletext.index(_CALL)
    build = moduletext.find(_BUILD, call)
    if not guard < call < build or 'StockTrack::IterationCapture capture(' not in moduletext[call:build]:
        raise ValueError('unsupported selected-target/native-build ordering')
    replacements = (
        (_INCLUDE, _INCLUDE + '#include "TrkFastSim/inc/CrossingGeometryWriter.hh"\n'),
        (_MEMBER, _MEMBER + '    CrossingGeometryWriter crossingGeometry_;\n'),
        (_BEGIN, _BEGIN + '    if (fullSchedule_) crossingGeometry_.beginJob(*tfs);\n'),
        (_CALL, _CALL + _INSERT),
    )
    result = moduletext
    for before, after in replacements:
        result = result.replace(before, after, 1)
    return result
