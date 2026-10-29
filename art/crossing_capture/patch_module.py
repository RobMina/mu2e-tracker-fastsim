"""Wire selected-only crossing diagnostics without changing original fit or capture fields."""
from patch_core import replace
import patch_geometry


def patched(text):
    text = patch_geometry.patch(text)
    text = replace(text, '#include "TrkFastSim/inc/IterationCaptureWriter.hh"',
                   '#include "TrkFastSim/inc/IterationCaptureWriter.hh"\n#include "TrkFastSim/inc/CrossingCaptureWriter.hh"\n#include "TrkFastSim/inc/CrossingCaptureStatus.hh"')
    text = replace(text, '    IterationCaptureWriter<KTRAJ> iterationWriter_;',
                   '    IterationCaptureWriter<KTRAJ> iterationWriter_;\n    CrossingCaptureWriter crossingWriter_;\n    CrossingCaptureStatus crossingStatus_;')
    text = replace(text, '    if (fullSchedule_) iterationWriter_.beginJob(*tfs);',
                   '    if (fullSchedule_) iterationWriter_.beginJob(*tfs);\n    if (fullSchedule_) { crossingWriter_.beginJob(*tfs); crossingStatus_.beginJob(*tfs); }')
    text = replace(text, '          StockTrack::IterationCapture capture(iterationCaptureRequested_ ? &iterations : nullptr, &captureHitOperands<KTRAJ>);',
                   '''          KinKal::CrossingTraceSink crossingSink;
          if(iterationCaptureRequested_){
            try { crossingStatus_.write(run_,subrun_,event_,iseed,0,crossingSink,""); }
            catch(...) { crossingSink.fail("Cannot write crossing attempt status"); }
          }
          auto crossingObserver = +[](KinKal::ElementXing<KTRAJ> const& x,long long iteration,long long effect,long long input){
            if(auto const* sx=dynamic_cast<KKStrawXing<KTRAJ> const*>(&x))sx->captureCrossing(3,nullptr,-1,iteration,effect,input);
          };
          StockTrack::IterationCapture capture(iterationCaptureRequested_ ? &iterations : nullptr, &captureHitOperands<KTRAJ>,
                                              iterationCaptureRequested_ ? &crossingSink : nullptr,crossingObserver);''')
    text = replace(text, '            iterationWriter_.write(*trk, iterations, iterationCaptureWritten_, iterationCaptureErrors_);',
                   '''            iterationWriter_.write(*trk, iterations, iterationCaptureWritten_, iterationCaptureErrors_);
            try {
              crossingWriter_.write(run_,subrun_,event_,iseed,crossingSink);
              crossingStatus_.write(run_,subrun_,event_,iseed,1,crossingSink,"");
            } catch(std::exception const& error) {
              try { crossingStatus_.write(run_,subrun_,event_,iseed,-1,crossingSink,error.what()); } catch(...) {}
            } catch(...) {
              try { crossingStatus_.write(run_,subrun_,event_,iseed,-1,crossingSink,"Unknown crossing serialization failure"); } catch(...) {}
            }''')
    text = replace(text, '    if (fullSchedule_) { crossingWriter_.beginJob(*tfs); crossingStatus_.beginJob(*tfs); }',
                   '    if (fullSchedule_) { try { crossingWriter_.beginJob(*tfs); crossingStatus_.beginJob(*tfs); } catch(...) {} }')
    text = replace(text, '    if (fullSchedule_) crossingGeometry_.beginJob(*tfs);',
                   '    if (fullSchedule_) { try { crossingGeometry_.beginJob(*tfs); } catch(...) {} }')
    return text
