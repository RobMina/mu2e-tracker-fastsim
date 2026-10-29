#ifndef Mu2eKinKal_CrossingCapture_hh
#define Mu2eKinKal_CrossingCapture_hh
#include "KinKal/Fit/Track.hh"
#include "KinKal/Fit/CrossingTrace.hh"
#include "KinKal/General/Parameters.hh"
#include <stdexcept>
#include <utility>

namespace mu2e {
  template<class V> std::vector<double> crossingXYZ(V const& v) { return {v.X(),v.Y(),v.Z()}; }
  template<class KTRAJ> KinKal::CrossingPieceTrace crossingPiece(KTRAJ const& trajectory) {
    KinKal::CrossingPieceTrace out;
    auto const& p = trajectory.params();
    for(size_t i=0; i<KinKal::NParams(); ++i){
      out.parameters.push_back(p.parameters()[i]);
      for(size_t j=0; j<KinKal::NParams(); ++j)out.covariance.push_back(p.covariance()[i][j]);
    }
    out.range = {trajectory.range().begin(),trajectory.range().end()};
    out.bnom = crossingXYZ(trajectory.bnom()); out.mass = trajectory.mass(); out.charge = trajectory.charge();
    return out;
  }
  template<class CA> KinKal::CrossingStateTrace crossingState(CA const& ca) {
    KinKal::CrossingStateTrace out;
    out.status = ca.status(); out.usable = ca.usable(); out.precision = ca.precision();
    out.reference = crossingPiece(ca.particleTraj());
    auto const& line = ca.sensorTraj();
    out.lineStart = crossingXYZ(line.start()); out.lineEnd = crossingXYZ(line.end());
    out.lineDirection = crossingXYZ(line.direction());
    out.lineMeasurementTime = line.measurementTime(); out.lineLength = line.length();
    if(out.usable){
      auto const& p = ca.particlePoca(); auto const& s = ca.sensorPoca();
      out.particlePoca = {p.X(),p.Y(),p.Z(),p.T()}; out.sensorPoca = {s.X(),s.Y(),s.Z(),s.T()};
      out.particleDirection = crossingXYZ(ca.particleDirection()); out.sensorDirection = crossingXYZ(ca.sensorDirection());
      out.doca = ca.doca(); out.docaVariance = ca.docaVar(); out.tocaVariance = ca.tocaVar(); out.lsign = ca.lSign();
      for(size_t i=0; i<KinKal::NParams(); ++i){ out.dDdP.push_back(ca.dDdP()[i]); out.dTdP.push_back(ca.dTdP()[i]); }
    }
    return out;
  }
  template<class KTRAJ> void crossingBeginTrajectory(KinKal::ParticleTrajectory<KTRAJ> const& trajectory, int category) noexcept {
    auto* sink = KinKal::Track<KTRAJ>::crossingCaptureSink();
    if(!sink)return;
    sink->copy([&]{
      KinKal::CrossingTrajectoryTrace out; out.category = category;
      for(auto const& piece : trajectory.pieces())out.pieces.push_back(crossingPiece(*piece));
      sink->trajectories.push_back(std::move(out));
      sink->currentTrajectory = sink->trajectories.size()-1;
    });
  }
  template<class KTRAJ, class XING> void crossingBindSearch(XING const& crossing, KinKal::PiecewiseCATrace const& search) noexcept {
    auto* sink = KinKal::Track<KTRAJ>::crossingCaptureSink();
    if(!sink)return;
    sink->copy([&]{
      if(search.failed)throw std::runtime_error("Crossing PCA trace failed");
      auto object = sink->lookup(&crossing);
      auto& record = sink->records.at(sink->constructors.at(object));
      if(record.hasSearch)throw std::runtime_error("Duplicate crossing constructor search");
      record.search = search; record.hasSearch = true;
    });
  }
  template<class KTRAJ, class COPY> void crossingDiscovery(int stage, COPY const& copy) noexcept {
    auto* sink = KinKal::Track<KTRAJ>::crossingCaptureSink();
    if(!sink)return;
    sink->copy([&]{
      KinKal::CrossingDiscoveryTrace out; out.stage = stage; out.trajectory = sink->currentTrajectory;
      copy(out);
      if(out.hasSearch && out.search.failed)throw std::runtime_error("Discovery PCA trace failed");
      sink->discovery.push_back(std::move(out));
    });
  }
}
#endif
