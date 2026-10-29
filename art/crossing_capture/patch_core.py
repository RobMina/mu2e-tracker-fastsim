"""Exact-anchor capture hooks; no native build or runtime action."""


def replace(text, before, after):
    if text.count(before) != 1:
        raise ValueError("crossing source anchor is not unique: "+before[:120])
    return text.replace(before, after, 1)


def track(text):
    text = replace(text, '#include "KinKal/Fit/IterationTrace.hh"', '#include "KinKal/Fit/IterationTrace.hh"\n#include "KinKal/Fit/CrossingTrace.hh"')
    text = replace(text, '      class IterationCapture {', '      using CrossingObserver = void (*)(EXING const&, long long, long long, long long);\n      class IterationCapture {')
    text = replace(text, 'explicit IterationCapture(std::vector<IterationTrace>* sink, HitOperandObserver observer=nullptr) noexcept :',
                   'explicit IterationCapture(std::vector<IterationTrace>* sink, HitOperandObserver observer=nullptr, CrossingTraceSink* crossings=nullptr, CrossingObserver crossingObserver=nullptr) noexcept :')
    text = replace(text, 'previousSink_(iterationSink_), previousCapture_(iterationCapture_), observer_(sink ? observer : nullptr)',
                   'previousSink_(iterationSink_), previousCapture_(iterationCapture_), observer_(sink ? observer : nullptr), crossings_(sink ? crossings : nullptr), crossingObserver_(sink ? crossingObserver : nullptr)')
    text = replace(text, '          HitOperandObserver observer_;', '          HitOperandObserver observer_;\n          CrossingTraceSink* crossings_;\n          CrossingObserver crossingObserver_;')
    text = replace(text, '      // construct from a set of hits and passive material crossings.',
                   '      static CrossingTraceSink* crossingCaptureSink() noexcept { return iterationCapture_ ? iterationCapture_->crossings_ : nullptr; }\n      // construct from a set of hits and passive material crossings.')
    text = replace(text, '            if(iterationCapture_)++iterationCapture_->attempted_;',
                   '            if(iterationCapture_)++iterationCapture_->attempted_;\n            if(auto* crossings=crossingCaptureSink())crossings->iteration = iterationCapture_->attempted_-1;')
    text = replace(text, '      void captureEffects(IterationTrace* trace, KKEFFFWDBND const* bounds) noexcept;',
                   '''      static void captureCrossingTrajectory(PKTRAJ const& trajectory, int category) noexcept {
        if(auto* sink=crossingCaptureSink())sink->copy([&]{
          CrossingTrajectoryTrace out; out.category=category;
          for(auto const& piece : trajectory.pieces())out.pieces.push_back(crossingPieceTrace(*piece));
          sink->trajectories.push_back(std::move(out)); sink->currentTrajectory=sink->trajectories.size()-1;
        });
      }
      void captureEffects(IterationTrace* trace, KKEFFFWDBND const* bounds) noexcept;''')
    text = replace(text, '    auto const& miconfig = firstMetaIter();\n    for(auto& hit : hits ) {',
                   '    auto const& miconfig = firstMetaIter();\n    captureCrossingTrajectory(*fittraj_,3);\n    for(auto& hit : hits ) {')
    text = replace(text, '      for(auto& effptr : effects_) effptr->updateReference(*ptraj,miconfig);',
                   '      captureCrossingTrajectory(*ptraj,4);\n      for(auto& effptr : effects_) effptr->updateReference(*ptraj,miconfig);')
    text = replace(text, '    for (auto& eff : effects_) {\n      eff->updateReference(*newtraj,firstMetaIter());',
                   '    captureCrossingTrajectory(*newtraj,5);\n    for (auto& eff : effects_) {\n      eff->updateReference(*newtraj,firstMetaIter());')
    text = replace(text, '    for(auto& exing : exings) {\n      effects_.emplace_back(std::make_unique<KKMAT>(exing,*fittraj_,miconfig));',
                   '    for(auto& exing : exings) {\n      CrossingReferenceStage crossingStage(crossingCaptureSink(),2);\n      effects_.emplace_back(std::make_unique<KKMAT>(exing,*fittraj_,miconfig));')
    text = replace(text, '          out.change = crossing->params();',
                   '          out.change = crossing->params();\n          if(iterationCapture_ && iterationCapture_->crossingObserver_)\n            iterationCapture_->crossingObserver_(*crossing,iterationCapture_->attempted_-1,trace->effects.size()-1,out.inputIndex);')
    return text


XING_METHODS = '''      void captureCrossing(int stage, KinKal::PiecewiseCATrace const* search=nullptr, int hintKind=-1,
                           long long iteration=-1, long long effect=-1, long long input=-1) const noexcept {
        auto* sink = KinKal::Track<KTRAJ>::crossingCaptureSink();
        if(!sink)return;
        sink->copy([&]{
          KinKal::CrossingRecord out;
          out.stage=stage; out.hintKind=hintKind; out.trajectory=sink->currentTrajectory;
          out.iteration=iteration < 0 ? sink->iteration : iteration; out.effect=effect; out.input=input;
          out.strawid=strawId().asUint16(); out.partner=shptr_ ? static_cast<long long>(shptr_->strawHitIndex()) : -1;
          out.offset=toff_; out.intrinsicActive=active_; out.active=active(); out.state=crossingState(ca_);
          if(ca_.usable())out.effectTime=time();
          if(sink->currentTrajectory < 0)throw std::runtime_error("Crossing trajectory was not captured");
          out.category=sink->trajectories.at(sink->currentTrajectory).category;
          if(stage==0){
            if(!sink->pendingOffset)throw std::runtime_error("Missing crossing offset operands");
            out.radiusOperand=sink->pendingRadius; out.speedOperand=sink->pendingSpeed;
            sink->pendingOffset=false;
            out.object=sink->identities.size();
            if(!sink->identities.emplace(this,out.object).second)throw std::runtime_error("Duplicate crossing constructor address");
            sink->constructors.push_back(sink->records.size());
          }else{
            out.object=sink->lookup(this);
            out.category=sink->records.at(sink->constructors.at(out.object)).category;
          }
          if(search){
            if(search->failed)throw std::runtime_error("Crossing reference PCA trace failed");
            out.search=*search; out.hasSearch=true;
          }
          sink->append(std::move(out));
        });
      }
'''
OFFSET_METHOD = '''      static double captureOffset(double radius, double speed) noexcept {
        if(auto* sink=KinKal::Track<KTRAJ>::crossingCaptureSink()){
          if(sink->pendingOffset)sink->fail("Unconsumed crossing offset operands");
          sink->pendingRadius=radius; sink->pendingSpeed=speed; sink->pendingOffset=true;
        }
        return radius/speed;
      }
'''


def xing(text):
    text = replace(text, '#include "Offline/Mu2eKinKal/inc/KKStrawHit.hh"',
                   '#include "Offline/Mu2eKinKal/inc/KKStrawHit.hh"\n#include "Offline/Mu2eKinKal/inc/CrossingCapture.hh"')
    text = replace(text, '      void setHit(KKSTRAWHITPTR& shptr) { shptr_ = shptr; }',
                   '      void setHit(KKSTRAWHITPTR& shptr) { shptr_ = shptr; captureCrossing(4); }\n'+XING_METHODS)
    text = replace(text, '      StrawXingUpdater sxconfig_;', OFFSET_METHOD+'      StrawXingUpdater sxconfig_;')
    text = replace(text, 'toff_(smat.wireRadius()/ca.particleTraj().speed(ca.particleToca()))',
                   'toff_(captureOffset(smat.wireRadius(),ca.particleTraj().speed(ca.particleToca())))')
    text = replace(text, 'toff_(smat.wireRadius()/strawhit->closestApproach().particleTraj().speed(strawhit->closestApproach().particleToca()))',
                   'toff_(captureOffset(smat.wireRadius(),strawhit->closestApproach().particleTraj().speed(strawhit->closestApproach().particleToca())))')
    before = '    varscale_(1.0)\n  {}'
    if text.count(before) != 2:
        raise ValueError("expected both crossing constructors")
    text = text.replace(before, '    varscale_(1.0)\n  { captureCrossing(0); }')
    text = replace(text, '    CAHint tphint(strawAxis()->timeAtMidpoint(),strawAxis()->timeAtMidpoint());',
                   '    CAHint tphint(strawAxis()->timeAtMidpoint(),strawAxis()->timeAtMidpoint());\n    int crossingHintKind=0;')
    text = replace(text, '      tphint = ca_.hint();', '      tphint = ca_.hint(); crossingHintKind=1;')
    text = replace(text, '      tphint = shptr_->closestApproach().hint();', '      tphint = shptr_->closestApproach().hint(); crossingHintKind=2;')
    text = replace(text, '    PCA pca(ptraj,strawAxis(),tphint,precision(),miconfig.xingPieceIter());',
                   '    KinKal::PiecewiseCATrace crossingSearch;\n    auto* crossingSink=KinKal::Track<KTRAJ>::crossingCaptureSink();\n    PCA pca(ptraj,strawAxis(),tphint,precision(),miconfig.xingPieceIter(),crossingSink ? &crossingSearch : nullptr);')
    text = replace(text, '    ca_ = static_cast<CA>(pca);',
                   '    ca_ = static_cast<CA>(pca);\n    if(crossingSink)captureCrossing(crossingSink->referenceStage,&crossingSearch,crossingHintKind);')
    return text
