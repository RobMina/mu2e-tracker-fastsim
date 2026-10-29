#ifndef TrkFastSim_CrossingCaptureWriter_hh
#define TrkFastSim_CrossingCaptureWriter_hh

#include "KinKal/Fit/CrossingTrace.hh"
#include "art_root_io/TFileService.h"
#include "TTree.h"
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace mu2e {
// Selected-capture sidecars only. Full target identity is (run,subrun,event,iseed).
// Stages: 0 constructor, 1 ordinary reference update, 2 wrapper reference update,
// 3 prefilter, 4 setHit; iteration/effect/input are required only at stage 3;
// radius_operand/speed_operand are constructor-only (NaN on subsequent stages).
// CA-derived scalars/vectors are valid only when ca_usable; search_* only when
// has_search; reference/line data are copied independently of CA usability.
// Piece parameters and row-major covariance are native q/C, without conversion.
// All ordinals are zero-based within the selected target. No pointer is persisted.
// Validate the entire sink before Fill; ROOT failures can still leave partial trees,
// so the caller must invalidate the capture on any exception. No physics is called.
class CrossingCaptureWriter {
  bool ready_ = false;
  TTree* records_ = nullptr;
  TTree* trajectories_ = nullptr;
  TTree* discovery_ = nullptr;
  int schema_ = 1, run_ = 0, subrun_ = 0, event_ = 0, iseed_ = 0;
  KinKal::CrossingRecord record_;
  KinKal::CrossingDiscoveryTrace discoveryRow_;
  KinKal::CrossingPieceTrace piece_;
  unsigned long long trajectoryIndex_ = 0, pieceIndex_ = 0, discoverySequence_ = 0;
  int trajectoryCategory_ = -1;

  static void require(bool ok, char const* message) {
    if (!ok) throw std::runtime_error(message);
  }
  template<class T> static void branch(TTree* tree, std::string const& name, T* value) {
    require(tree && tree->Branch(name.c_str(), value), "Crossing capture ROOT Branch failed");
  }
  static void fill(TTree* tree) {
    require(tree && tree->Fill() >= 0, "Crossing capture ROOT Fill failed");
  }
  void common(TTree* tree) {
    branch(tree, "schema_version", &schema_);
    branch(tree, "run", &run_); branch(tree, "subrun", &subrun_);
    branch(tree, "event", &event_); branch(tree, "iseed", &iseed_);
  }
  static void pieceBranches(TTree* tree, std::string const& prefix, KinKal::CrossingPieceTrace& p) {
    branch(tree, prefix + "parameters", &p.parameters);
    branch(tree, prefix + "covariance", &p.covariance);
    branch(tree, prefix + "range", &p.range);
    branch(tree, prefix + "bnom", &p.bnom);
    branch(tree, prefix + "mass", &p.mass);
    branch(tree, prefix + "charge", &p.charge);
  }
  static void stateBranches(TTree* tree, KinKal::CrossingStateTrace& s) {
    branch(tree, "ca_status", &s.status); branch(tree, "ca_usable", &s.usable);
    branch(tree, "precision", &s.precision);
    branch(tree, "doca", &s.doca); branch(tree, "doca_variance", &s.docaVariance);
    branch(tree, "toca_variance", &s.tocaVariance); branch(tree, "lsign", &s.lsign);
    branch(tree, "particle_poca", &s.particlePoca); branch(tree, "sensor_poca", &s.sensorPoca);
    branch(tree, "particle_direction", &s.particleDirection);
    branch(tree, "sensor_direction", &s.sensorDirection);
    branch(tree, "d_dd_p", &s.dDdP); branch(tree, "d_td_p", &s.dTdP);
    branch(tree, "line_start", &s.lineStart); branch(tree, "line_end", &s.lineEnd);
    branch(tree, "line_direction", &s.lineDirection);
    branch(tree, "line_measurement_time", &s.lineMeasurementTime);
    branch(tree, "line_length", &s.lineLength);
    pieceBranches(tree, "reference_", s.reference);
  }
  static void searchBranches(TTree* tree, KinKal::PiecewiseCATrace& s) {
    branch(tree, "search_failed", &s.failed); branch(tree, "search_exhausted", &s.exhausted);
    branch(tree, "search_initial_index", &s.initialIndex);
    branch(tree, "search_final_index", &s.finalIndex);
    branch(tree, "search_solution_index", &s.solutionIndex);
    branch(tree, "search_max_moves", &s.maxMoves);
    branch(tree, "search_particle_hint", &s.particleHint);
    branch(tree, "search_sensor_hint", &s.sensorHint);
    branch(tree, "search_precision", &s.precision);
    branch(tree, "search_solution_indices", &s.solutionIndices);
    branch(tree, "search_input_particle_hints", &s.inputParticleHints);
    branch(tree, "search_input_sensor_hints", &s.inputSensorHints);
    branch(tree, "search_particle_times", &s.particleTimes);
    branch(tree, "search_sensor_times", &s.sensorTimes);
    branch(tree, "search_statuses", &s.statuses);
    branch(tree, "search_usable", &s.usable);
  }
  static void validateSearch(KinKal::PiecewiseCATrace const& s) {
    require(!s.failed, "Crossing capture contains failed PCA search");
    auto const n = s.solutionIndices.size();
    require(n != 0 && s.inputParticleHints.size() == n && s.inputSensorHints.size() == n &&
            s.particleTimes.size() == n && s.sensorTimes.size() == n &&
            s.statuses.size() == n && s.usable.size() == n,
            "Crossing capture has missing or misaligned PCA search columns");
    // Unusable/exhausted solutions are legitimate diagnostics, not capture failures.
  }
  static KinKal::CrossingStateTrace copyState(KinKal::CrossingStateTrace const& s) {
    KinKal::CrossingStateTrace out;
    out.status = s.status; out.usable = s.usable; out.precision = s.precision;
    out.reference = s.reference;
    out.lineStart = s.lineStart; out.lineEnd = s.lineEnd; out.lineDirection = s.lineDirection;
    out.lineMeasurementTime = s.lineMeasurementTime; out.lineLength = s.lineLength;
    // Guard before accessing CA-derived values; defaults are NaN / empty vectors.
    if (out.usable) {
      out.doca = s.doca; out.docaVariance = s.docaVariance;
      out.tocaVariance = s.tocaVariance; out.lsign = s.lsign;
      out.particlePoca = s.particlePoca; out.sensorPoca = s.sensorPoca;
      out.particleDirection = s.particleDirection; out.sensorDirection = s.sensorDirection;
      out.dDdP = s.dDdP; out.dTdP = s.dTdP;
    }
    return out;
  }
  static void validate(KinKal::CrossingTraceSink const& sink) {
    if (sink.failed) throw std::runtime_error(sink.error ? sink.error : "Crossing capture sink failed");
    require(!sink.pendingOffset, "Crossing capture has unconsumed constructor offset operands");
    auto const objects = sink.constructors.size();
    require(sink.identities.size() == objects, "Crossing capture constructor identity count differs");
    std::vector<bool> identities(objects, false);
    for (auto const& identity : sink.identities) {
      require(identity.second < objects && !identities[identity.second],
              "Crossing capture identity map is not one-to-one");
      identities[identity.second] = true;
    }
    std::vector<bool> seen(objects, false);
    std::size_t nextObject = 0;
    std::set<std::tuple<long long, long long>> prefilterKeys;
    for (std::size_t i = 0; i < sink.records.size(); ++i) {
      auto const& r = sink.records[i];
      require(r.sequence == i, "Crossing capture record sequence is not contiguous");
      require(r.stage >= 0 && r.stage <= 4, "Crossing capture has unknown record stage");
      require(r.object < objects, "Crossing capture record has no constructor identity");
      require(r.category >= 0 && r.strawid >= 0, "Crossing capture record identity is missing");
      require(std::isfinite(r.offset) && r.offset > 0, "Crossing capture stored offset is invalid");
      // Persist the stored offset verbatim: do not reevaluate radius/speed here.
      if (r.trajectory >= 0)
        require(static_cast<unsigned long long>(r.trajectory) < sink.trajectories.size(),
                "Crossing capture trajectory identity is out of range");
      if (r.stage == 0) {
        require(std::isfinite(r.radiusOperand) && r.radiusOperand > 0 &&
                std::isfinite(r.speedOperand) && r.speedOperand > 0,
                "Crossing capture native constructor offset operands are invalid");
        require(r.object == nextObject && !seen[r.object] && sink.constructors[r.object] == i,
                "Crossing capture constructor identities are duplicate or noncontiguous");
        require(r.hasSearch, "Crossing capture constructor search was not bound");
        require(r.trajectory >= 0, "Crossing capture constructor has no trajectory");
        require(sink.trajectories[static_cast<std::size_t>(r.trajectory)].category == r.category,
                "Crossing capture constructor trajectory category differs");
        seen[r.object] = true; ++nextObject;
      } else {
        require(seen[r.object], "Crossing capture update precedes its constructor");
        auto const& ctor = sink.records[sink.constructors[r.object]];
        require(r.category == ctor.category && r.strawid == ctor.strawid,
                "Crossing capture update identity differs from constructor");
      }
      if (r.stage == 1 || r.stage == 2)
        require(r.hasSearch, "Crossing capture reference search is missing");
      if (r.stage == 3) {
        require(r.iteration >= 0 && r.effect >= 0 && r.input >= 0,
                "Crossing capture prefilter iteration/effect/input is missing");
        require(prefilterKeys.emplace(r.iteration, r.effect).second,
                "Crossing capture duplicate prefilter iteration/effect key");
      }
      if (r.hasSearch) validateSearch(r.search);
    }
    require(nextObject == objects, "Crossing capture is missing constructor records");
    for (auto const& trajectory : sink.trajectories)
      require(!trajectory.pieces.empty(), "Crossing capture trajectory has no pieces");
    for (auto const& d : sink.discovery) {
      require(d.stage >= 0, "Crossing capture discovery stage is missing");
      if (d.trajectory >= 0)
        require(static_cast<unsigned long long>(d.trajectory) < sink.trajectories.size(),
                "Crossing capture discovery trajectory is out of range");
      if (d.object >= 0)
        require(static_cast<unsigned long long>(d.object) < objects,
                "Crossing capture discovery object has no constructor");
      if (d.hasSearch) validateSearch(d.search);
    }
  }

public:
  CrossingCaptureWriter() = default;
  CrossingCaptureWriter(CrossingCaptureWriter const&) = delete;
  CrossingCaptureWriter& operator=(CrossingCaptureWriter const&) = delete;

  void beginJob(art::TFileService& fs) {
    require(!records_ && !trajectories_ && !discovery_, "Crossing capture beginJob called twice");
    records_ = fs.make<TTree>("crossing_records", "Selected crossing constructor, reference, prefilter and setHit records");
    trajectories_ = fs.make<TTree>("crossing_trajectories", "Selected crossing native piecewise trajectories");
    discovery_ = fs.make<TTree>("crossing_discovery", "Selected crossing ordered discovery decisions");
    common(records_); common(trajectories_); common(discovery_);
    branch(records_, "object", &record_.object); branch(records_, "sequence", &record_.sequence);
    branch(records_, "stage", &record_.stage); branch(records_, "category", &record_.category);
    branch(records_, "hint_kind", &record_.hintKind); branch(records_, "strawid", &record_.strawid);
    branch(records_, "partner", &record_.partner);
    branch(records_, "trajectory_index", &record_.trajectory);
    branch(records_, "iteration_ordinal", &record_.iteration);
    branch(records_, "effect_index", &record_.effect); branch(records_, "input_index", &record_.input);
    branch(records_, "intrinsic_active", &record_.intrinsicActive); branch(records_, "active", &record_.active);
    branch(records_, "has_search", &record_.hasSearch); branch(records_, "offset", &record_.offset);
    branch(records_, "effect_time", &record_.effectTime);
    branch(records_, "radius_operand", &record_.radiusOperand); branch(records_, "speed_operand", &record_.speedOperand);
    stateBranches(records_, record_.state); searchBranches(records_, record_.search);
    branch(trajectories_, "trajectory_index", &trajectoryIndex_);
    branch(trajectories_, "category", &trajectoryCategory_);
    branch(trajectories_, "piece_index", &pieceIndex_);
    pieceBranches(trajectories_, "", piece_);
    branch(discovery_, "sequence", &discoverySequence_);
    branch(discovery_, "stage", &discoveryRow_.stage);
    branch(discovery_, "trajectory_index", &discoveryRow_.trajectory);
    branch(discovery_, "plane", &discoveryRow_.plane); branch(discovery_, "panel", &discoveryRow_.panel);
    branch(discovery_, "straw", &discoveryRow_.straw); branch(discovery_, "object", &discoveryRow_.object);
    branch(discovery_, "passed", &discoveryRow_.passed);
    branch(discovery_, "integers", &discoveryRow_.integers); branch(discovery_, "values", &discoveryRow_.values);
    branch(discovery_, "has_search", &discoveryRow_.hasSearch);
    stateBranches(discovery_, discoveryRow_.state); searchBranches(discovery_, discoveryRow_.search);
    ready_ = true;
  }

  void write(int run, int subrun, int event, int iseed, KinKal::CrossingTraceSink const& sink) {
    require(ready_ && records_ && trajectories_ && discovery_, "Crossing capture writer is not initialized");
    validate(sink);
    run_ = run; subrun_ = subrun; event_ = event; iseed_ = iseed;
    trajectoryIndex_ = 0;
    for (auto const& trajectory : sink.trajectories) {
      trajectoryCategory_ = trajectory.category; pieceIndex_ = 0;
      for (auto const& piece : trajectory.pieces) {
        piece_ = piece; fill(trajectories_); ++pieceIndex_;
      }
      ++trajectoryIndex_;
    }
    for (auto const& r : sink.records) {
      // Explicit value copies avoid even accessing invalid CA-derived payloads.
      record_.object = r.object; record_.sequence = r.sequence;
      record_.stage = r.stage; record_.category = r.category; record_.hintKind = r.hintKind;
      record_.strawid = r.strawid; record_.partner = r.partner; record_.trajectory = r.trajectory;
      record_.iteration = r.iteration; record_.effect = r.effect; record_.input = r.input;
      record_.intrinsicActive = r.intrinsicActive; record_.active = r.active;
      record_.hasSearch = r.hasSearch; record_.offset = r.offset; record_.effectTime = r.effectTime;
      record_.radiusOperand = r.radiusOperand; record_.speedOperand = r.speedOperand;
      record_.state = copyState(r.state);
      record_.search = r.hasSearch ? r.search : KinKal::PiecewiseCATrace{};
      fill(records_);
    }
    discoverySequence_ = 0;
    for (auto const& d : sink.discovery) {
      discoveryRow_.stage = d.stage; discoveryRow_.trajectory = d.trajectory;
      discoveryRow_.plane = d.plane; discoveryRow_.panel = d.panel;
      discoveryRow_.straw = d.straw; discoveryRow_.object = d.object;
      discoveryRow_.passed = d.passed; discoveryRow_.integers = d.integers; discoveryRow_.values = d.values;
      discoveryRow_.hasSearch = d.hasSearch; discoveryRow_.state = copyState(d.state);
      discoveryRow_.search = d.hasSearch ? d.search : KinKal::PiecewiseCATrace{};
      fill(discovery_); ++discoverySequence_;
    }
  }
};
} // namespace mu2e
#endif
