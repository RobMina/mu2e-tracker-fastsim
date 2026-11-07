// Stock driftInfo queries for validation, not an interpolated reconstruction conditions model.
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "TTree.h"
#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Principal/Event.h"
#include "art_root_io/TFileService.h"
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"
#include "fhiclcpp/types/Sequence.h"
#include "Offline/BFieldGeom/inc/BFieldManager.hh"
#include "Offline/DbTables/inc/DbIoV.hh"
#include "Offline/GeometryService/inc/DetectorSystem.hh"
#include "Offline/GeometryService/inc/GeomHandle.hh"
#include "Offline/ProditionsService/inc/ProditionsHandle.hh"
#include "Offline/TrackerConditions/inc/StrawResponse.hh"
#include "Offline/TrackerGeom/inc/Tracker.hh"

namespace mu2e {
class FieldOnDriftOracle : public art::EDAnalyzer {
 public:
  struct Config {
    using Name = fhicl::Name;
    using Comment = fhicl::Comment;
    fhicl::Atom<double> timeMin{Name("TimeMin"), Comment("oracle only, ns"), -50.0};
    fhicl::Atom<double> timeMax{Name("TimeMax"), Comment("oracle only, ns"), 200.0};
    fhicl::Atom<double> timeStep{Name("TimeStep"), Comment("rectangular oracle grid, ns"), 0.05};
    fhicl::Atom<unsigned> angleBins{Name("AngleBins"), Comment("inclusive 0..pi/2 oracle grid"), 77};
    fhicl::Atom<unsigned> extraProbes{Name("ExtraProbes"), Comment("deterministic off-grid probes"), 8192};
    fhicl::Atom<unsigned> maxQueries{Name("MaxQueries"), Comment("raise, never truncate"), 1000000};
    fhicl::Sequence<double> timeProbes{Name("TimeProbes"), Comment("extra paired dt probes, ns"), std::vector<double>{}};
    fhicl::Sequence<double> angleProbes{Name("AngleProbes"), Comment("extra paired phi probes, rad"), std::vector<double>{}};
  };
  using Parameters = art::EDAnalyzer::Table<Config>;
  explicit FieldOnDriftOracle(Parameters const& c);
  void beginJob() override;
  void analyze(art::Event const& event) override;
 private:
  void addQuery(StrawResponse const& response, StrawId sid, double time, double phi);
  ProditionsHandle<StrawResponse> response_;
  ProditionsHandle<Tracker> tracker_;
  double timeMin_, timeMax_, timeStep_;
  unsigned angleBins_, extraProbes_, maxQueries_;
  size_t timeBins_;
  std::vector<double> timeProbes_, angleProbes_;
  bool done_{false};
  DbIoV iov_;
  TTree* tree_{nullptr};
  unsigned run_{0}, subrun_{0}, event_{0}, strawId_{0};
  unsigned startRun_{0}, startSubrun_{0}, endRun_{0}, endSubrun_{0};
  int schema_{1};
  std::string scope_{"stock_query_oracle_only_not_interpolation_conditions"};
  std::vector<int> responseCids_, trackerCids_;
  std::vector<double> origin_, field_, dt_, phi_, usedPhi_, cluster_, radius_, velocity_, signedError_, unsignedError_;
  double cc_{0}, wireRadius_{0}, outerRadius_{0}, roundedBz_{0};
};

FieldOnDriftOracle::FieldOnDriftOracle(Parameters const& c)
    : art::EDAnalyzer{c}, timeMin_(c().timeMin()), timeMax_(c().timeMax()), timeStep_(c().timeStep()),
      angleBins_(c().angleBins()), extraProbes_(c().extraProbes()), maxQueries_(c().maxQueries()),
      timeProbes_(c().timeProbes()), angleProbes_(c().angleProbes()) {
  if (!std::isfinite(timeMin_) || !std::isfinite(timeMax_) || !std::isfinite(timeStep_)
      || timeStep_ <= 0 || timeMin_ >= timeMax_ || angleBins_ < 2 || maxQueries_ == 0 || maxQueries_ > 1000000
      || timeProbes_.size() != angleProbes_.size())
    throw cet::exception("DRIFTORACLE_CONFIG") << "Invalid finite query grid or paired probes";
  double intervals = (timeMax_ - timeMin_) / timeStep_;
  if (!std::isfinite(intervals) || intervals > maxQueries_
      || std::abs(intervals - std::round(intervals)) > 1e-8)
    throw cet::exception("DRIFTORACLE_CONFIG") << "Time extent must have an integral bounded interval count";
  timeBins_ = static_cast<size_t>(std::round(intervals)) + 1;
  if (timeBins_ * static_cast<size_t>(angleBins_) + extraProbes_ + timeProbes_.size() > maxQueries_)
    throw cet::exception("DRIFTORACLE_CONFIG") << "Query bound exceeded; no grid truncation";
  for (size_t i = 0; i < timeProbes_.size(); ++i)
    if (!std::isfinite(timeProbes_[i]) || !std::isfinite(angleProbes_[i]))
      throw cet::exception("DRIFTORACLE_CONFIG") << "Nonfinite extra probe";
}

void FieldOnDriftOracle::beginJob() {
  art::ServiceHandle<art::TFileService> tfs;
  tree_ = tfs->make<TTree>("oracle", "Stock public driftInfo query oracle, first conditions IOV only");
  tree_->Branch("schema", &schema_);
  tree_->Branch("scope", &scope_);
  for (auto [name, value] : std::vector<std::pair<const char*, unsigned*>>{
           {"run", &run_}, {"subrun", &subrun_}, {"event", &event_}, {"reference_straw", &strawId_},
           {"iov_start_run", &startRun_}, {"iov_start_subrun", &startSubrun_},
           {"iov_end_run", &endRun_}, {"iov_end_subrun", &endSubrun_}, {"angle_bins", &angleBins_},
           {"extra_probes", &extraProbes_}})
    tree_->Branch(name, value);
  for (auto [name, value] : std::vector<std::pair<const char*, double*>>{
           {"cc", &cc_}, {"wire_radius", &wireRadius_}, {"outer_radius", &outerRadius_},
           {"origin_bz_float", &roundedBz_}, {"time_min", &timeMin_}, {"time_max", &timeMax_}, {"time_step", &timeStep_}})
    tree_->Branch(name, value);
  for (auto [name, value] : std::vector<std::pair<const char*, std::vector<double>*>>{
           {"detector_origin_mu2e", &origin_}, {"origin_field", &field_}, {"dt", &dt_}, {"phi", &phi_},
           {"used_phi", &usedPhi_}, {"c_drift", &cluster_}, {"r_drift", &radius_}, {"drift_velocity", &velocity_},
           {"signed_error", &signedError_}, {"unsigned_error", &unsignedError_}})
    tree_->Branch(name, value);
  tree_->Branch("response_cids", &responseCids_);
  tree_->Branch("tracker_cids", &trackerCids_);
}

void FieldOnDriftOracle::addQuery(StrawResponse const& response, StrawId sid, double time, double phi) {
  auto info = response.driftInfo(sid, time, phi);
  for (double value : {info.LorentzAngle_, info.cDrift_, info.rDrift_, info.driftVelocity_,
                       info.signedDriftError_, info.unsignedDriftError_})
    if (!std::isfinite(value)) throw cet::exception("DRIFTORACLE_VALUE") << "Nonfinite stock driftInfo";
  dt_.push_back(time); phi_.push_back(phi); usedPhi_.push_back(info.LorentzAngle_);
  cluster_.push_back(info.cDrift_); radius_.push_back(info.rDrift_); velocity_.push_back(info.driftVelocity_);
  signedError_.push_back(info.signedDriftError_); unsignedError_.push_back(info.unsignedDriftError_);
}

void FieldOnDriftOracle::analyze(art::Event const& event) {
  if (done_) {
    if (!iov_.inInterval(event.run(), event.subRun()))
      throw cet::exception("DRIFTORACLE_IOV") << "Event outside exported conditions intersection";
    return;
  }
  auto const& response = response_.get(event.id());
  auto const& tracker = tracker_.get(event.id());
  if (tracker.getStraws().empty()) throw cet::exception("DRIFTORACLE_GEOMETRY") << "Empty tracker";
  iov_ = response_.iov();
  iov_.overlap(tracker_.iov());
  if (iov_.isNull() || !iov_.inInterval(event.run(), event.subRun()))
    throw cet::exception("DRIFTORACLE_IOV") << "Invalid first-event conditions intersection";
  startRun_ = iov_.startRun(); startSubrun_ = iov_.startSubrun();
  endRun_ = iov_.endRun(); endSubrun_ = iov_.endSubrun();
  responseCids_.assign(response.getCids().begin(), response.getCids().end());
  trackerCids_.assign(tracker.getCids().begin(), tracker.getCids().end());
  run_ = event.run(); subrun_ = event.subRun(); event_ = event.event();
  auto sid = tracker.getStraws().front().id();
  strawId_ = sid.asUint16();
  GeomHandle<DetectorSystem> detector;
  GeomHandle<BFieldManager> field;
  GeomHandle<Tracker> nominalTracker;
  auto origin = detector->toMu2e(CLHEP::Hep3Vector(0., 0., 0.));
  CLHEP::Hep3Vector b;
  if (!field->getBFieldWithStatus(origin, b))
    throw cet::exception("DRIFTORACLE_FIELD") << "No field coverage at detector origin";
  origin_ = {origin.x(), origin.y(), origin.z()};
  field_ = {b.x(), b.y(), b.z()};
  for (auto value : field_)
    if (!std::isfinite(value)) throw cet::exception("DRIFTORACLE_FIELD") << "Nonfinite origin field";
  roundedBz_ = static_cast<float>(b.z());
  cc_ = response.strawDrift().GetCC();
  wireRadius_ = nominalTracker->wireRadius(); outerRadius_ = nominalTracker->strawOuterRadius();
  for (auto value : origin_)
    if (!std::isfinite(value)) throw cet::exception("DRIFTORACLE_GEOMETRY") << "Nonfinite detector origin";
  if (!std::isfinite(cc_) || !std::isfinite(roundedBz_) || !std::isfinite(wireRadius_)
      || !std::isfinite(outerRadius_) || wireRadius_ <= 0 || outerRadius_ <= wireRadius_)
    throw cet::exception("DRIFTORACLE_GEOMETRY") << "Invalid drift field or nominal radii";
  double halfPi = std::acos(-1.) / 2.;
  for (size_t it = 0; it < timeBins_; ++it)
    for (unsigned ip = 0; ip < angleBins_; ++ip)
      addQuery(response, sid, timeMin_ + timeStep_ * it, halfPi * ip / (angleBins_ - 1));
  for (unsigned i = 0; i < extraProbes_; ++i) {
    double ft = std::fmod((i + 0.5) * 0.6180339887498948, 1.);
    double fp = std::fmod((i + 0.5) * 0.4142135623730950, 1.);
    addQuery(response, sid, timeMin_ + (timeMax_ - timeMin_) * ft, (8. * fp - 4.) * halfPi);
  }
  for (size_t i = 0; i < timeProbes_.size(); ++i) addQuery(response, sid, timeProbes_[i], angleProbes_[i]);
  tree_->Fill();
  done_ = true;
}
}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::FieldOnDriftOracle)
