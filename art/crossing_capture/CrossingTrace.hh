#ifndef KinKal_CrossingTrace_hh
#define KinKal_CrossingTrace_hh
#include "KinKal/Trajectory/PiecewiseCATrace.hh"
#include "KinKal/General/Parameters.hh"
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace KinKal {
  struct CrossingPieceTrace {
    std::vector<double> parameters, covariance, range, bnom;
    double mass = std::numeric_limits<double>::quiet_NaN();
    int charge = 0;
  };

  template<class KTRAJ> CrossingPieceTrace crossingPieceTrace(KTRAJ const& trajectory) {
    CrossingPieceTrace out;
    auto const& p=trajectory.params();
    for(size_t i=0; i<NParams(); ++i){
      out.parameters.push_back(p.parameters()[i]);
      for(size_t j=0; j<NParams(); ++j)out.covariance.push_back(p.covariance()[i][j]);
    }
    out.range={trajectory.range().begin(),trajectory.range().end()};
    auto const& b=trajectory.bnom(); out.bnom={b.X(),b.Y(),b.Z()};
    out.mass=trajectory.mass(); out.charge=trajectory.charge();
    return out;
  }

  struct CrossingStateTrace {
    int status = 6;
    bool usable = false;
    double precision = std::numeric_limits<double>::quiet_NaN();
    double doca = std::numeric_limits<double>::quiet_NaN(), docaVariance = std::numeric_limits<double>::quiet_NaN();
    double tocaVariance = std::numeric_limits<double>::quiet_NaN(), lsign = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> particlePoca, sensorPoca, particleDirection, sensorDirection, dDdP, dTdP;
    std::vector<double> lineStart, lineEnd, lineDirection;
    double lineMeasurementTime = std::numeric_limits<double>::quiet_NaN(), lineLength = std::numeric_limits<double>::quiet_NaN();
    CrossingPieceTrace reference;
  };

  struct CrossingRecord {
    unsigned long long object = 0, sequence = 0;
    int stage = -1, category = -1, hintKind = -1;
    long long strawid = -1, partner = -1, trajectory = -1, iteration = -1, effect = -1, input = -1;
    bool intrinsicActive = false, active = false, hasSearch = false;
    double offset = std::numeric_limits<double>::quiet_NaN(), effectTime = std::numeric_limits<double>::quiet_NaN();
    double radiusOperand = std::numeric_limits<double>::quiet_NaN(), speedOperand = std::numeric_limits<double>::quiet_NaN();
    CrossingStateTrace state;
    PiecewiseCATrace search;
  };

  struct CrossingTrajectoryTrace {
    int category = -1;
    std::vector<CrossingPieceTrace> pieces;
  };

  struct CrossingDiscoveryTrace {
    int stage = -1;
    long long trajectory = -1, plane = -1, panel = -1, straw = -1, object = -1;
    bool passed = false;
    std::vector<long long> integers;
    std::vector<double> values;
    bool hasSearch = false;
    PiecewiseCATrace search;
    CrossingStateTrace state;
  };

  struct CrossingTraceSink {
    bool failed = false;
    char const* error = nullptr;
    int referenceStage = 1;
    long long currentTrajectory = -1;
    long long iteration = -1;
    double pendingRadius = std::numeric_limits<double>::quiet_NaN(), pendingSpeed = std::numeric_limits<double>::quiet_NaN();
    bool pendingOffset = false;
    std::vector<CrossingRecord> records;
    std::vector<CrossingTrajectoryTrace> trajectories;
    std::vector<CrossingDiscoveryTrace> discovery;
    std::map<void const*, size_t> identities;
    std::vector<size_t> constructors;
    void fail(char const* message) noexcept { failed = true; if(!error)error = message; }
    template<class COPY> void copy(COPY const& callback) noexcept {
      if(failed)return;
      try { callback(); } catch(...) { fail("Crossing diagnostic copy failed"); }
    }
    size_t lookup(void const* ptr) const {
      auto it = identities.find(ptr);
      if(it == identities.end())throw std::runtime_error("Crossing has no captured constructor");
      return it->second;
    }
    void append(CrossingRecord row) {
      row.sequence = records.size();
      records.push_back(std::move(row));
    }
  };

  class CrossingReferenceStage {
    CrossingTraceSink* sink_;
    int previous_;
  public:
    CrossingReferenceStage(CrossingTraceSink* sink, int stage) noexcept : sink_(sink), previous_(sink ? sink->referenceStage : 1) {
      if(sink_)sink_->referenceStage = stage;
    }
    ~CrossingReferenceStage() noexcept { if(sink_)sink_->referenceStage = previous_; }
    CrossingReferenceStage(CrossingReferenceStage const&) = delete;
    CrossingReferenceStage& operator=(CrossingReferenceStage const&) = delete;
  };
}
#endif
