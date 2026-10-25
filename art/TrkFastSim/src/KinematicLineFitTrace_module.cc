// Stage-by-stage trace of the stock KinematicLineFit (KKLine) on every CosmicTrackSeed: the fit is rerun with the
// stock schedule truncated after each meta-iteration (seed fit stages 0..3; extension stages 4..12 on the full seed
// fit), so the state after stage k equals stock's state after meta-iteration k (the fit is deterministic). One TTree
// entry per (seed, stage): fit status, chi2, hit states and qualities, crossing activity, front/back momentum.
// Same configuration tables as KinematicLineFit; no products.
#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Run.h"
#include "art/Framework/Principal/Handle.h"
#include "art_root_io/TFileService.h"
#include "fhiclcpp/types/Atom.h"
#include "fhiclcpp/types/Sequence.h"
#include "fhiclcpp/types/Table.h"
#include "fhiclcpp/types/OptionalTable.h"
#include "fhiclcpp/types/OptionalAtom.h"
#include "Offline/GlobalConstantsService/inc/GlobalConstantsHandle.hh"
#include "Offline/GlobalConstantsService/inc/ParticleDataList.hh"
#include "Offline/ProditionsService/inc/ProditionsHandle.hh"
#include "Offline/TrackerConditions/inc/StrawResponse.hh"
#include "Offline/TrackerConditions/inc/TrackerStatus.hh"
#include "Offline/BFieldGeom/inc/BFieldManager.hh"
#include "Offline/GeometryService/inc/GeomHandle.hh"
#include "Offline/GeometryService/inc/DetectorSystem.hh"
#include "Offline/TrackerGeom/inc/Tracker.hh"
#include "Offline/CalorimeterGeom/inc/Calorimeter.hh"
#include "Offline/RecoDataProducts/inc/CaloCluster.hh"
#include "Offline/RecoDataProducts/inc/ComboHit.hh"
#include "Offline/RecoDataProducts/inc/CosmicTrackSeed.hh"
#include "Offline/DataProducts/inc/PDGCode.hh"
#include "KinKal/Fit/Track.hh"
#include "KinKal/Fit/Config.hh"
#include "KinKal/Trajectory/KinematicLine.hh"
#include "KinKal/Trajectory/ParticleTrajectory.hh"
#include "Offline/Mu2eKinKal/inc/KKFit.hh"
#include "Offline/Mu2eKinKal/inc/KKFitSettings.hh"
#include "Offline/Mu2eKinKal/inc/KKStrawHit.hh"
#include "Offline/Mu2eKinKal/inc/KKBField.hh"
#include "Offline/Mu2eKinKal/inc/KKFitUtilities.hh"
#include "TTree.h"
#include <memory>
#include <vector>

namespace mu2e {
  using KTRAJ = KinKal::KinematicLine;
  using PTRAJ = KinKal::ParticleTrajectory<KTRAJ>;
  using KKTRK = KKTrack<KTRAJ>;
  using KKSTRAWHITCOL = std::vector<std::shared_ptr<KKStrawHit<KTRAJ>>>;
  using KKSTRAWXINGCOL = std::vector<std::shared_ptr<KKStrawXing<KTRAJ>>>;
  using KKCALOHITCOL = std::vector<std::shared_ptr<KKCaloHit<KTRAJ>>>;
  using PARAMHITCOL = std::vector<std::shared_ptr<KinKal::ParameterHit<KTRAJ>>>;
  using KKFIT = KKFit<KTRAJ>;
  using KinKal::Config;
  using KinKal::DMAT;
  using KinKal::VEC3;
  using StrawHitIndexCollection = std::vector<StrawHitIndex>;

  class KinematicLineFitTrace : public art::EDAnalyzer {
    using Name = fhicl::Name;
    using Comment = fhicl::Comment;
    struct KKLineModuleConfig : public Mu2eKinKal::KKModuleConfig {
      fhicl::Sequence<art::InputTag> seedCollections {Name("CosmicTrackSeedCollections"), Comment("seeds")};
      fhicl::Atom<float> seedmom {Name("SeedMomentum"), Comment("Initial momentum value")};
      fhicl::Sequence<double> paramconstraints {Name("ParameterConstraints"), Comment("parameter constraint sigmas")};
      fhicl::Sequence<std::string> sampleSurfaces {Name("SampleSurfaces"), Comment("unused here")};
      fhicl::Atom<bool> sampleInRange {Name("SampleInRange"), Comment("unused here")};
      fhicl::Atom<bool> sampleInBounds {Name("SampleInBounds"), Comment("unused here")};
      fhicl::Atom<float> interTol {Name("IntersectionTolerance"), Comment("unused here")};
      fhicl::Atom<float> sampleTBuff {Name("SampleTimeBuffer"), Comment("unused here")};
    };
    struct GlobalConfig {
      fhicl::Table<KKLineModuleConfig> modSettings {Name("ModuleSettings")};
      fhicl::Table<Mu2eKinKal::KKFitConfig> mu2eSettings {Name("KKFitSettings")};
      fhicl::Table<Mu2eKinKal::KinKalConfig> fitSettings {Name("FitSettings")};
      fhicl::Table<Mu2eKinKal::KinKalConfig> extSettings {Name("ExtensionSettings")};
      fhicl::OptionalTable<Mu2eKinKal::KKExtrapConfig> extrapSettings {Name("ExtrapolationSettings")};   // accepted, unused
      fhicl::OptionalAtom<std::string> fitDirection {Name("FitDirection"), Comment("unused here")};
      fhicl::Atom<int> maxSeeds {Name("MaxSeedsPerEvent"), Comment("trace at most this many seeds per event"), 4};
      fhicl::Atom<bool> perIter {Name("PerIteration"), Comment("diagnostic caps applied to every retained meta-iteration"), false};
      fhicl::Atom<bool> fullSchedule {Name("FullSchedule"), Comment("one full seed fit and extension without stage reruns"), false};
      fhicl::Atom<bool> captureRejection {Name("CaptureRejection"), Comment("capture the exact rejected extension candidate"), false};
    };

  public:
    using Parameters = art::EDAnalyzer::Table<GlobalConfig>;
    explicit KinematicLineFitTrace(Parameters const& settings);
    void beginJob() override;
    void beginRun(art::Run const& run) override;
    void analyze(art::Event const& event) override;

  private:
    KTRAJ makeSeedTraj(CosmicTrackSeed const& hseed) const;
    bool goodFit(KKTRK const& ktrk) const;
    void record(KKTRK const& trk, int iseed, int stage, bool goodseed);
    std::vector<art::ProductToken<CosmicTrackSeedCollection>> seedCols_;
    art::ProductToken<ComboHitCollection> chcol_T_;
    TrkFitFlag goodline_;
    ProditionsHandle<StrawResponse> strawResponse_h_;
    ProditionsHandle<Tracker> alignedTracker_h_;
    float seedmom_;
    std::vector<double> paramconstraints_;
    PDGCode::type fpart_;
    KKFIT kkfit_;
    DMAT seedcov_;
    bool constraining_ = false;
    double mass_ = 0;
    int charge_ = 0;
    int maxseeds_;
    bool periter_, fullSchedule_, captureRejection_;
    int rejected_ = 0;
    std::vector<double> rejectedParameters_, rejectedCovariances_, rejectedRanges_, rejectedSteps_, rejectedMetrics_;
    std::vector<unsigned long long> rejectedReferenceIndices_;
    std::unique_ptr<KKBField> kkbf_;
    Config config_, exconfig_;
    std::vector<Config> seedcfgs_, extcfgs_;
    // output
    TTree* tr_ = nullptr;
    int run_ = 0, subrun_ = 0, event_ = 0, iseed_ = 0, stage_ = 0, status_ = 0, miter_ = 0, iter_ = 0, ndof_ = 0;
    int goodseed_ = 0, nxact_ = 0, nxing_ = 0, maxit_ = -1;
    std::vector<double> pf_, pb_, cf_, cb_;          // Retained front/back parameters and covariance.
    std::vector<int> ihist_, sthist_;               // Full status history: meta-iteration, iteration, status.
    std::vector<double> chist_;
    double chi2_ = 0, momf_ = 0, momb_ = 0, momfvar_ = 0;
    std::vector<int> shind_, state_, frozen_, algo_;
    std::vector<float> qbkg_, qsign_, qdrift_, doca_;
  };

  KinematicLineFitTrace::KinematicLineFitTrace(Parameters const& settings) : art::EDAnalyzer{settings},
    chcol_T_(consumes<ComboHitCollection>(settings().modSettings().comboHitCollection())),
    goodline_(settings().modSettings().seedFlags()),
    seedmom_(settings().modSettings().seedmom()),
    paramconstraints_(settings().modSettings().paramconstraints()),
    fpart_(static_cast<PDGCode::type>(settings().modSettings().fitParticle())),
    kkfit_(settings().mu2eSettings()),
    maxseeds_(settings().maxSeeds()),
    periter_(settings().perIter()),
    fullSchedule_(settings().fullSchedule()),
    captureRejection_(settings().captureRejection()),
    config_(Mu2eKinKal::makeConfig(settings().fitSettings())),
    exconfig_(Mu2eKinKal::makeConfig(settings().extSettings())) {
    if ((captureRejection_ && !fullSchedule_) || (fullSchedule_ && periter_))
      throw std::invalid_argument("CaptureRejection requires FullSchedule; FullSchedule forbids PerIteration");
    for (auto const& tag : settings().modSettings().seedCollections())
      seedCols_.emplace_back(consumes<CosmicTrackSeedCollection>(tag));
    auto const& seederrors = settings().modSettings().seederrors();
    for (size_t i = 0; i < seederrors.size() && i < KinKal::NParams(); ++i) seedcov_[i][i] = seederrors[i] * seederrors[i];
    for (auto c : paramconstraints_) constraining_ |= c > 0;
    // truncated schedules: stage k runs meta-iterations 0..k of its configuration
    for (size_t k = 1; k <= config_.schedule().size(); ++k) {
      Config c = config_;
      c.schedule_.resize(k);
      seedcfgs_.push_back(c);
    }
    for (size_t k = 1; k <= exconfig_.schedule().size(); ++k) {
      Config c = exconfig_;
      c.schedule_.resize(k);
      extcfgs_.push_back(c);
    }
  }

  void KinematicLineFitTrace::beginJob() {
    art::ServiceHandle<art::TFileService> tfs;
    tr_ = tfs->make<TTree>("trace", "KinematicLineFit stage-by-stage trace");
    tr_->Branch("run", &run_); tr_->Branch("subrun", &subrun_); tr_->Branch("event", &event_);
    tr_->Branch("iseed", &iseed_); tr_->Branch("stage", &stage_); tr_->Branch("goodseed", &goodseed_);
    tr_->Branch("status", &status_); tr_->Branch("miter", &miter_); tr_->Branch("iter", &iter_);
    tr_->Branch("chi2", &chi2_); tr_->Branch("ndof", &ndof_);
    tr_->Branch("momf", &momf_); tr_->Branch("momb", &momb_); tr_->Branch("momfvar", &momfvar_);
    tr_->Branch("nxing", &nxing_); tr_->Branch("nxact", &nxact_);
    tr_->Branch("shind", &shind_); tr_->Branch("state", &state_); tr_->Branch("frozen", &frozen_);
    tr_->Branch("algo", &algo_); tr_->Branch("qbkg", &qbkg_); tr_->Branch("qsign", &qsign_);
    tr_->Branch("qdrift", &qdrift_); tr_->Branch("doca", &doca_);
    tr_->Branch("maxit", &maxit_); tr_->Branch("pf", &pf_); tr_->Branch("pb", &pb_); tr_->Branch("cf", &cf_);
    tr_->Branch("cb", &cb_); tr_->Branch("hmiter", &ihist_); tr_->Branch("hstatus", &sthist_); tr_->Branch("hchi2", &chist_);
    tr_->Branch("capture_enabled", &captureRejection_); tr_->Branch("full_schedule", &fullSchedule_);
    tr_->Branch("rejected", &rejected_); tr_->Branch("rejected_parameters", &rejectedParameters_);
    tr_->Branch("rejected_covariances", &rejectedCovariances_); tr_->Branch("rejected_ranges", &rejectedRanges_);
    tr_->Branch("rejected_steps", &rejectedSteps_); tr_->Branch("rejected_metrics", &rejectedMetrics_);
    tr_->Branch("rejected_reference_indices", &rejectedReferenceIndices_);
  }

  void KinematicLineFitTrace::beginRun(art::Run const&) {
    auto const& ptable = GlobalConstantsHandle<ParticleDataList>();
    mass_ = ptable->particle(fpart_).mass();
    charge_ = static_cast<int>(ptable->particle(fpart_).charge());
    GeomHandle<BFieldManager> bfmgr;
    GeomHandle<DetectorSystem> det;
    kkbf_ = std::make_unique<KKBField>(*bfmgr, *det);
  }

  KTRAJ KinematicLineFitTrace::makeSeedTraj(CosmicTrackSeed const& hseed) const {
    VEC3 bnom(0.0, 0.0, 0.001);
    KinKal::VEC4 pos(hseed._track.MinuitParams.A0, 0, hseed._track.MinuitParams.B0, hseed._t0._t0);
    XYZVectorF mom3(hseed._track.MinuitParams.A1, -1, hseed._track.MinuitParams.B1);
    mom3 = mom3.Unit() * seedmom_;
    KinKal::MOM4 mom(mom3.x(), mom3.y(), mom3.z(), mass_);
    auto seedtraj = KTRAJ(pos, mom, charge_, bnom, Mu2eKinKal::timeBounds(hseed.hits()));
    seedtraj.params().covariance() = seedcov_;
    return seedtraj;
  }

  bool KinematicLineFitTrace::goodFit(KKTRK const& ktrk) const {
    if (!ktrk.fitStatus().usable()) return false;
    for (auto const& sh : ktrk.strawHits())
      if (sh->active() && !Mu2eKinKal::inDetector(ktrk.fitTraj().position3(sh->time()))) return false;
    return true;
  }

  void KinematicLineFitTrace::record(KKTRK const& trk, int iseed, int stage, bool goodseed) {
    iseed_ = iseed; stage_ = stage; goodseed_ = goodseed;
    rejected_ = 0;
    for (auto* v : {&rejectedParameters_, &rejectedCovariances_, &rejectedRanges_, &rejectedSteps_, &rejectedMetrics_}) v->clear();
    rejectedReferenceIndices_.clear();
    if (auto const* r = trk.rejectionTrace()) {
      rejected_ = 1;
      for (auto const* piece : {&r->candidateFront, &r->referenceFront, &r->candidateBack, &r->referenceBack}) {
        rejectedRanges_.push_back(piece->range().begin()); rejectedRanges_.push_back(piece->range().end());
        for (size_t i = 0; i < KinKal::NParams(); ++i) {
          rejectedParameters_.push_back(piece->params().parameters()[i]);
          for (size_t j = 0; j < KinKal::NParams(); ++j)
            rejectedCovariances_.push_back(piece->params().covariance()(i, j));
        }
      }
      for (auto const* step : {&r->frontStep, &r->backStep})
        for (size_t i = 0; i < KinKal::NParams(); ++i) rejectedSteps_.push_back((*step)[i]);
      rejectedReferenceIndices_ = {r->referenceFrontIndex, r->referenceBackIndex};
      rejectedMetrics_ = {r->frontChi2, r->backChi2, r->threshold, r->averageGap, r->deltaChi2};
    }
    auto const& st = trk.fitStatus();
    status_ = st.status_; miter_ = st.miter_; iter_ = st.iter_; chi2_ = st.chisq_.chisq(); ndof_ = st.chisq_.nDOF();
    auto const& front = trk.fitTraj().front();
    momf_ = front.mom(); momfvar_ = front.params().covariance()(KTRAJ::mom_, KTRAJ::mom_);
    momb_ = trk.fitTraj().back().mom();
    auto const& back = trk.fitTraj().back();
    for (auto* v : {&pf_, &pb_, &cf_, &cb_, &chist_}) v->clear();
    ihist_.clear(); sthist_.clear();
    for (size_t i = 0; i < KinKal::NParams(); ++i) {
      pf_.push_back(front.params().parameters()[i]); pb_.push_back(back.params().parameters()[i]);
      for (size_t j = 0; j < KinKal::NParams(); ++j) {
        cf_.push_back(front.params().covariance()(i, j)); cb_.push_back(back.params().covariance()(i, j));
      }
    }
    for (auto const& h : trk.history()) {
      ihist_.push_back(1000 * h.miter_ + h.iter_); sthist_.push_back(h.status_); chist_.push_back(h.chisq_.chisq());
    }
    for (auto* v : {&shind_, &state_, &frozen_, &algo_}) v->clear();
    for (auto* v : {&qbkg_, &qsign_, &qdrift_, &doca_}) v->clear();
    for (auto const& sh : trk.strawHits()) {
      auto const& w = sh->hitState();
      shind_.push_back(sh->strawHitIndex()); state_.push_back(w.state_); frozen_.push_back(w.frozen_);
      algo_.push_back(w.algo_);
      qbkg_.push_back(w.quality(WireHitState::bkg)); qsign_.push_back(w.quality(WireHitState::sign));
      qdrift_.push_back(w.quality(WireHitState::drift));
      doca_.push_back(sh->closestApproach().usable() ? sh->closestApproach().doca() : -999.f);
    }
    nxing_ = trk.strawXings().size();
    nxact_ = 0;
    for (auto const& x : trk.strawXings()) nxact_ += x->active();
    tr_->Fill();
  }

  void KinematicLineFitTrace::analyze(art::Event const& event) {
    run_ = event.run(); subrun_ = event.subRun(); event_ = event.event();
    auto const& strawresponse = strawResponse_h_.getPtr(event.id());
    auto const& tracker = alignedTracker_h_.getPtr(event.id()).get();
    GeomHandle<Calorimeter> calo_h;
    auto const& chcol = *event.getValidHandle<ComboHitCollection>(chcol_T_);
    art::Handle<CaloClusterCollection> cc_H;   // invalid: KKFIT UseCaloCluster must be false
    int iseed = -1;
    for (auto const& tag : seedCols_) {
      auto const& seeds = *event.getValidHandle<CosmicTrackSeedCollection>(tag);
      for (auto const& hseed : seeds) {
        ++iseed;
        if (iseed >= maxseeds_ || !hseed.status().hasAllProperties(goodline_)) continue;
        auto build = [&](Config const& cfg) {
          KTRAJ seedtraj = makeSeedTraj(hseed);
          PTRAJ pseedtraj(seedtraj);
          StrawHitIndexCollection idx;
          auto chcolptr = hseed.hits().fillStrawHitIndices(idx, StrawIdMask::uniquestraw);
          KKSTRAWHITCOL sh;
          KKSTRAWXINGCOL sx;
          kkfit_.makeStrawHits(*tracker, *strawresponse, *kkbf_, pseedtraj, *chcolptr, idx, sh, sx);
          KKCALOHITCOL ch;
          seedtraj.range() = kkfit_.range(sh, ch, sx);
          PARAMHITCOL ph;
          if (constraining_) kkfit_.makeSeedParamHit(seedtraj, paramconstraints_, ph);
          return std::make_unique<KKTRK>(cfg, *kkbf_, seedtraj, fpart_, kkfit_.strawHitClusterer(), sh, sx, ch, ph);
        };
        if (fullSchedule_) {
          maxit_ = -1;
          auto trk = build(config_);
          bool const seedgood = goodFit(*trk);
          if (seedgood) {
            trk->captureRejection(captureRejection_);
            kkfit_.extendTrack(exconfig_, *kkbf_, *tracker, *strawresponse, chcol, *calo_h, cc_H, *trk);
          }
          record(*trk, iseed, -1, seedgood);
          continue;
        }
        for (size_t k = 0; k < seedcfgs_.size(); ++k) {
          maxit_ = -1;
          auto trk = build(seedcfgs_[k]);
          record(*trk, iseed, k, true);
        }
        auto full = build(config_);
        if (!goodFit(*full)) continue;   // stock: no extension
        for (size_t k = 0; k < extcfgs_.size(); ++k) {
          maxit_ = -1;
          auto trk = build(config_);
          kkfit_.extendTrack(extcfgs_[k], *kkbf_, *tracker, *strawresponse, chcol, *calo_h, cc_H, *trk);
          record(*trk, iseed, seedcfgs_.size() + k, true);
          // Capped reruns can change every retained meta-iteration; they are not successive stock snapshots.
          if (!periter_) continue;
          unsigned nit = 0;
          unsigned const lastmeta = trk->fitStatus().miter_;
          for (auto const& h : trk->history())
            if (h.miter_ == lastmeta) nit = std::max(nit, h.iter_ + 1);
          for (unsigned m = 1; m < nit; ++m) {
            Config c = extcfgs_[k];
            c.maxniter_ = m;   // applies to every meta-iteration of the extension: only valid if the earlier ones
                               // converged within m iterations; recorded per row so the reader can check
            auto t2 = build(config_);
            kkfit_.extendTrack(c, *kkbf_, *tracker, *strawresponse, chcol, *calo_h, cc_H, *t2);
            maxit_ = static_cast<int>(m);
            record(*t2, iseed, seedcfgs_.size() + k, true);
          }
        }
      }
    }
  }
}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::KinematicLineFitTrace)
