"""Add source-ordered constructor and normal-branch discovery observations without changing cuts."""
import re
from patch_core import replace


def constructors(text):
    text = replace(text, '    unsigned ngood(0);\n    GeomHandle<mu2e::KKMaterial> kkmat_h;',
                   '    unsigned ngood(0);\n    crossingBeginTrajectory(ptraj,0);\n    GeomHandle<mu2e::KKMaterial> kkmat_h;')
    text = replace(text, '    KKSTRAWHITCOL addstrawhits;',
                   '    crossingBeginTrajectory(kktrk.fitTraj(),1);\n    KKSTRAWHITCOL addstrawhits;')
    text = replace(text, '    // pre-compute some tracker info if needed\n    if(needstrackerinfo_)fillTrackerInfo(tracker);',
                   '    crossingBeginTrajectory(ptraj,2);\n    // pre-compute some tracker info if needed\n    if(needstrackerinfo_)fillTrackerInfo(tracker);')
    pattern = r'(?m)^(\s*)PCA spca\(ptraj, sline, hint, tprec_ \);'
    found = list(re.finditer(pattern, text))
    if len(found) != 3:
        raise ValueError("expected three initial/associated PCA constructors")
    text = re.sub(pattern, lambda m: m[1]+'KinKal::PiecewiseCATrace crossingSearch;\n'+m[1]
                  +'PCA spca(ptraj, sline, hint, tprec_, 10, KinKal::Track<KTRAJ>::crossingCaptureSink() ? &crossingSearch : nullptr);', text)
    for collection, cast in (("exings", "CA const&"), ("addexings", "CA")):
        line = f'{collection}.push_back(std::make_shared<KKSTRAWXING>({"hits" if collection == "exings" else "addhits"}.back(),static_cast<{cast}>(spca),smat,straw));'
        if text.count(line) != (1 if collection == "exings" else 2):
            raise ValueError("associated crossing append differs")
        text = text.replace(line, line+f'\n                    crossingBindSearch<KTRAJ>(*{collection}.back(),crossingSearch);')
    return text


def material(text):
    start = text.index('  template <class KTRAJ> void KKFit<KTRAJ>::addStraws(')
    end = text.index('  template <class KTRAJ> void KKFit<KTRAJ>::addCaloHit(', start)
    block = text[start:end]
    block = replace(block, '    if (fabs(t0dir.Z()) < addStrawMinDz_){', '''    crossingDiscovery<KTRAJ>(0,[&](auto& row){
      row.passed=!(fabs(t0dir.Z()) < addStrawMinDz_);
      row.values={t0pos.X(),t0pos.Y(),t0pos.Z(),t0dir.X(),t0dir.Y(),t0dir.Z(),addStrawMinDz_,
                  strawradius_,rmin_,rmax_,ymin_,ymax_,spitch_,maxStrawDoca_,maxStrawDocaCon_,maxStrawUposBuff_,static_cast<double>(strawNBuffer_)};
      row.integers={static_cast<long long>(maxDStraw_),static_cast<long long>(skipStrawCheck_)};
      for(auto const& id : oldstraws)row.integers.push_back(id.asUint16());
    });
    if (fabs(t0dir.Z()) < addStrawMinDz_){
      if(auto* sink=KinKal::Track<KTRAJ>::crossingCaptureSink())sink->fail("Vertical material discovery not instrumented");''')
    marker = '      // Go hierarchically through planes and panels to find new straws hit by this track.'
    before, normal = block.split(marker)
    normal = replace(normal, '      for(auto const& plane : tracker.planes()){\n        if(tracker.planeExists(plane.id())) {', '''      for(auto const& plane : tracker.planes()){
        crossingDiscovery<KTRAJ>(1,[&](auto& row){row.plane=plane.id().asUint16(); row.passed=tracker.planeExists(plane.id());});
        if(tracker.planeExists(plane.id())) {''')
    normal = replace(normal, '          double rho = plpos.Rho();', '''          double rho = plpos.Rho();
          crossingDiscovery<KTRAJ>(2,[&](auto& row){
            row.plane=plane.id().asUint16(); row.values={plz,zt,plpos.X(),plpos.Y(),plpos.Z(),rho};
            row.passed=(rho > rmin_ && rho < rmax_) || skipStrawCheck_;
          });''')
    normal = replace(normal, '              int istraw = static_cast<int>(rint( (pposv.y()-ymin_)*spitch_));', '''              int istraw = static_cast<int>(rint( (pposv.y()-ymin_)*spitch_));
              crossingDiscovery<KTRAJ>(3,[&](auto& row){
                row.plane=plane.id().asUint16(); row.panel=panel.id().asUint16();
                row.values={tdir.X(),tdir.Y(),tdir.Z(),dz,papos.X(),papos.Y(),papos.Z(),pposv.x(),pposv.y(),pposv.z(),dirv};
                row.integers={ibuffer,istraw,static_cast<long long>(panel.nStraws())};
                row.passed=istraw >= -maxDStraw_-ibuffer && istraw < static_cast<int>(panel.nStraws()) + maxDStraw_+ibuffer;
              });''')
    normal = replace(normal, '                unsigned istrmin = static_cast<unsigned>(std::max(istraw-maxDStraw_-ibuffer,0));', '''                unsigned istrmin = static_cast<unsigned>(std::max(istraw-maxDStraw_-ibuffer,0));
                crossingDiscovery<KTRAJ>(4,[&](auto& row){
                  row.plane=plane.id().asUint16(); row.panel=panel.id().asUint16(); row.integers={istrmin};
                  row.values={panel.getStraw(istrmin).halfLength()};
                  row.passed=(fabs(pposv.x()) < panel.getStraw(istrmin).halfLength() + maxStrawUposBuff_) || skipStrawCheck_;
                });''')
    normal = replace(normal, '                  unsigned istrmax = static_cast<unsigned>(std::min(istraw+maxDStraw_+ibuffer,static_cast<int>(panel.nStraws())-1));', '''                  unsigned istrmax = static_cast<unsigned>(std::min(istraw+maxDStraw_+ibuffer,static_cast<int>(panel.nStraws())-1));
                  crossingDiscovery<KTRAJ>(5,[&](auto& row){
                    row.plane=plane.id().asUint16(); row.panel=panel.id().asUint16(); row.integers={istrmin,istrmax}; row.passed=true;
                  });''')
    normal = replace(normal, '                    if(oldstraws.find(straw.id()) == oldstraws.end()){', '''                    crossingDiscovery<KTRAJ>(6,[&](auto& row){
                      row.plane=plane.id().asUint16(); row.panel=panel.id().asUint16(); row.straw=straw.id().asUint16();
                      row.integers={istr}; row.passed=oldstraws.find(straw.id()) == oldstraws.end();
                    });
                    if(oldstraws.find(straw.id()) == oldstraws.end()){''')
    normal = replace(normal, '                      PCA pca(ptraj, sline, hint, tprec_ );', '''                      KinKal::PiecewiseCATrace crossingSearch;
                      PCA pca(ptraj, sline, hint, tprec_, 10, KinKal::Track<KTRAJ>::crossingCaptureSink() ? &crossingSearch : nullptr);''')
    normal = replace(normal, '                      if(doca < maxStrawDoca_ && dsig < maxStrawDocaCon_ && du < straw.halfLength() + maxStrawUposBuff_){', '''                      crossingDiscovery<KTRAJ>(7,[&](auto& row){
                        row.plane=plane.id().asUint16(); row.panel=panel.id().asUint16(); row.straw=straw.id().asUint16();
                        row.values={du,doca,dsig,pca.docaVar(),straw.halfLength()}; row.search=crossingSearch; row.hasSearch=true; row.state=crossingState(pca);
                        row.passed=doca < maxStrawDoca_ && dsig < maxStrawDocaCon_ && du < straw.halfLength() + maxStrawUposBuff_;
                      });
                      if(doca < maxStrawDoca_ && dsig < maxStrawDocaCon_ && du < straw.halfLength() + maxStrawUposBuff_){''')
    normal = replace(normal, '                        addexings.push_back(std::make_shared<KKSTRAWXING>(shptr,static_cast<CA>(pca),smat,straw));', '''                        addexings.push_back(std::make_shared<KKSTRAWXING>(shptr,static_cast<CA>(pca),smat,straw));
                        crossingBindSearch<KTRAJ>(*addexings.back(),crossingSearch);
                        crossingDiscovery<KTRAJ>(8,[&](auto& row){
                          row.plane=plane.id().asUint16(); row.panel=panel.id().asUint16(); row.straw=straw.id().asUint16();
                          row.object=KinKal::Track<KTRAJ>::crossingCaptureSink()->lookup(addexings.back().get()); row.passed=true;
                        });''')
    return text[:start]+before+marker+normal+text[end:]


def patched(text):
    if 'crossingBeginTrajectory' in text:
        raise ValueError("crossing fit instrumentation already present")
    return material(constructors(text))
