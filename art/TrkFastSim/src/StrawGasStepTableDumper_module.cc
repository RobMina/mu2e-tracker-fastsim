// Dump the tracker truth of a dts file to flat ROOT TTrees: the reference tables for mu2e-tracker-fastsim.
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "TTree.h"
#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/SubRun.h"
#include "art_root_io/TFileService.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/types/Atom.h"

#include "Offline/MCDataProducts/inc/CosmicLivetime.hh"
#include "Offline/MCDataProducts/inc/GenParticle.hh"
#include "Offline/MCDataProducts/inc/MCTrajectory.hh"
#include "Offline/MCDataProducts/inc/PrimaryParticle.hh"
#include "Offline/MCDataProducts/inc/SimParticle.hh"
#include "Offline/MCDataProducts/inc/StepPointMC.hh"
#include "Offline/MCDataProducts/inc/StrawGasStep.hh"
#include "Offline/MCDataProducts/inc/SurfaceStep.hh"

namespace mu2e {

class StrawGasStepTableDumper : public art::EDAnalyzer {
 public:
  struct Config {
    fhicl::Atom<art::InputTag> strawGasSteps{fhicl::Name("strawGasSteps"), art::InputTag("compressDetStepMCs")};
    fhicl::Atom<art::InputTag> surfaceSteps{fhicl::Name("surfaceSteps"), art::InputTag("compressDetStepMCs")};
    fhicl::Atom<art::InputTag> mcTrajectories{fhicl::Name("mcTrajectories"), art::InputTag("compressDetStepMCs")};
    fhicl::Atom<art::InputTag> vdSteps{fhicl::Name("vdSteps"), art::InputTag("compressDetStepMCs:virtualdetector")};
    fhicl::Atom<art::InputTag> simParticles{fhicl::Name("simParticles"), art::InputTag("compressDetStepMCs")};
    fhicl::Atom<art::InputTag> primaryParticle{fhicl::Name("primaryParticle"), art::InputTag("FindMCPrimary")};
    fhicl::Atom<art::InputTag> genParticles{fhicl::Name("genParticles"), art::InputTag("cosmicTimeOffset")};
    fhicl::Atom<art::InputTag> livetime{fhicl::Name("livetime"), art::InputTag("generate")};
    fhicl::Atom<bool> writeSteps{fhicl::Name("writeSteps"), true};          // false: gen + subruns only (gen stream)
    fhicl::Atom<bool> writeTrajectories{fhicl::Name("writeTrajectories"), true};
  };
  using Parameters = art::EDAnalyzer::Table<Config>;
  explicit StrawGasStepTableDumper(const Parameters& conf);
  void beginJob() override;
  void analyze(const art::Event& event) override;
  void endSubRun(const art::SubRun& subrun) override;

 private:
  void ids(TTree* t);
  template <class T> void branches(TTree* t, std::vector<std::pair<const char*, T*>> const& v) {
    for (auto const& [n, p] : v) t->Branch(n, p);
  }
  Config conf_;
  TTree *sgs_{nullptr}, *ss_{nullptr}, *traj_{nullptr}, *vd_{nullptr}, *parts_{nullptr}, *gens_{nullptr},
      *subruns_{nullptr};
  unsigned run_{0}, subrun_{0}, event_{0};
  // StrawGasStep row (positions in the Mu2e frame, as stored)
  int nsgs_{0}, sid_{0}, plane_{0}, panel_{0}, straw_{0}, upanel_{0}, shape_{0}, ion_{0}, sp_{0};
  float eion_{0}, len_{0}, width_{0}, x0_{0}, y0_{0}, z0_{0}, x1_{0}, y1_{0}, z1_{0}, px_{0}, py_{0}, pz_{0};
  double t_{0};
  // SurfaceStep row (positions in DETECTOR coordinates, as stored)
  int surf_{0}, sindex_{0}, ssp_{0};
  float sedep_{0}, sx0_{0}, sy0_{0}, sz0_{0}, sx1_{0}, sy1_{0}, sz1_{0}, spx_{0}, spy_{0}, spz_{0};
  double st_{0};
  // MCTrajectory point row (Mu2e frame)
  int tsp_{0}, tpt_{0};
  float tx_{0}, ty_{0}, tz_{0}, tt_{0}, tke_{0};
  // virtual-detector StepPointMC row (Mu2e frame)
  int vdid_{0}, vsp_{0};
  float vx_{0}, vy_{0}, vz_{0}, vpx_{0}, vpy_{0}, vpz_{0};
  double vt_{0};
  // SimParticle row (same columns as crvfastsim's CrvStepTableDumper, plus isMCPrimary)
  int key_{0}, parent_{0}, pdg_{0}, ccode_{0}, scode_{0}, isprim_{0}, ismcprim_{0}, genidx_{0};
  double psx_{0}, psy_{0}, psz_{0}, pex_{0}, pey_{0}, pez_{0};
  float pspx_{0}, pspy_{0}, pspz_{0}, pse_{0}, pepx_{0}, pepy_{0}, pepz_{0}, pee_{0}, pst_{0}, pet_{0};
  // GenParticle row
  int gidx_{0}, gpdg_{0}, ggen_{0};
  double gx_{0}, gy_{0}, gz_{0}, gpx_{0}, gpy_{0}, gpz_{0}, ge_{0}, gt_{0};
  // subrun row
  double livetime_{0};
  unsigned long long primaries_{0};
};

StrawGasStepTableDumper::StrawGasStepTableDumper(const Parameters& conf) : art::EDAnalyzer(conf), conf_(conf()) {}

void StrawGasStepTableDumper::ids(TTree* t) {
  t->Branch("run", &run_);
  t->Branch("subrun", &subrun_);
  t->Branch("event", &event_);
}

void StrawGasStepTableDumper::beginJob() {
  art::ServiceHandle<art::TFileService> tfs;
  gens_ = tfs->make<TTree>("gen", "GenParticles (every event)");
  ids(gens_);
  branches<int>(gens_, {{"idx", &gidx_}, {"pdg", &gpdg_}, {"genid", &ggen_}});
  branches<double>(gens_, {{"x", &gx_}, {"y", &gy_}, {"z", &gz_}, {"px", &gpx_}, {"py", &gpy_}, {"pz", &gpz_},
                           {"e", &ge_}, {"t", &gt_}});
  subruns_ = tfs->make<TTree>("subruns", "CosmicLivetime per subrun");
  subruns_->Branch("run", &run_);
  subruns_->Branch("subrun", &subrun_);
  subruns_->Branch("livetime", &livetime_);
  subruns_->Branch("primaries", &primaries_);
  if (!conf_.writeSteps()) return;

  sgs_ = tfs->make<TTree>("sgsteps", "StrawGasSteps (Mu2e frame)");
  ids(sgs_);
  branches<int>(sgs_, {{"nsgs", &nsgs_}, {"sid", &sid_}, {"plane", &plane_}, {"panel", &panel_}, {"straw", &straw_},
                       {"upanel", &upanel_}, {"shape", &shape_}, {"ion", &ion_}, {"sp", &sp_}});
  branches<float>(sgs_, {{"eion", &eion_}, {"len", &len_}, {"width", &width_}, {"x0", &x0_}, {"y0", &y0_},
                         {"z0", &z0_}, {"x1", &x1_}, {"y1", &y1_}, {"z1", &z1_}, {"px", &px_}, {"py", &py_},
                         {"pz", &pz_}});
  sgs_->Branch("t", &t_);

  ss_ = tfs->make<TTree>("surfsteps", "SurfaceSteps (detector frame)");
  ids(ss_);
  branches<int>(ss_, {{"surf", &surf_}, {"sindex", &sindex_}, {"sp", &ssp_}});
  branches<float>(ss_, {{"edep", &sedep_}, {"x0", &sx0_}, {"y0", &sy0_}, {"z0", &sz0_}, {"x1", &sx1_},
                        {"y1", &sy1_}, {"z1", &sz1_}, {"px", &spx_}, {"py", &spy_}, {"pz", &spz_}});
  ss_->Branch("t", &st_);

  vd_ = tfs->make<TTree>("vdsteps", "virtual-detector StepPointMCs (Mu2e frame)");
  ids(vd_);
  branches<int>(vd_, {{"vdid", &vdid_}, {"sp", &vsp_}});
  branches<float>(vd_, {{"x", &vx_}, {"y", &vy_}, {"z", &vz_}, {"px", &vpx_}, {"py", &vpy_}, {"pz", &vpz_}});
  vd_->Branch("t", &vt_);

  if (conf_.writeTrajectories()) {
    traj_ = tfs->make<TTree>("trajpts", "MCTrajectory points (Mu2e frame)");
    ids(traj_);
    branches<int>(traj_, {{"sp", &tsp_}, {"ipt", &tpt_}});
    branches<float>(traj_, {{"x", &tx_}, {"y", &ty_}, {"z", &tz_}, {"t", &tt_}, {"ke", &tke_}});
  }

  parts_ = tfs->make<TTree>("particles", "SimParticles");
  ids(parts_);
  branches<int>(parts_, {{"key", &key_}, {"parent", &parent_}, {"pdg", &pdg_}, {"ccode", &ccode_},
                         {"scode", &scode_}, {"isprim", &isprim_}, {"ismcprim", &ismcprim_}, {"genidx", &genidx_}});
  branches<double>(parts_, {{"sx", &psx_}, {"sy", &psy_}, {"sz", &psz_}, {"ex", &pex_}, {"ey", &pey_},
                            {"ez", &pez_}});
  branches<float>(parts_, {{"spx", &pspx_}, {"spy", &pspy_}, {"spz", &pspz_}, {"se", &pse_}, {"epx", &pepx_},
                           {"epy", &pepy_}, {"epz", &pepz_}, {"ee", &pee_}, {"st", &pst_}, {"et", &pet_}});
}

void StrawGasStepTableDumper::analyze(const art::Event& event) {
  run_ = event.run();
  subrun_ = event.subRun();
  event_ = event.event();
  auto gh = event.getValidHandle<GenParticleCollection>(conf_.genParticles());
  for (size_t i = 0; i < gh->size(); ++i) {
    auto const& g = gh->at(i);
    gidx_ = i;
    gpdg_ = g.pdgId();
    ggen_ = g.generatorId().id();
    gx_ = g.position().x();
    gy_ = g.position().y();
    gz_ = g.position().z();
    gpx_ = g.momentum().px();
    gpy_ = g.momentum().py();
    gpz_ = g.momentum().pz();
    ge_ = g.momentum().e();
    gt_ = g.time();
    gens_->Fill();
  }
  if (!conf_.writeSteps()) return;

  auto sh = event.getValidHandle<StrawGasStepCollection>(conf_.strawGasSteps());
  nsgs_ = sh->size();
  for (auto const& s : *sh) {
    sid_ = s.strawId().asUint16();
    plane_ = s.strawId().plane();
    panel_ = s.strawId().panel();
    straw_ = s.strawId().straw();
    upanel_ = s.strawId().uniquePanel();
    shape_ = s.stepType().shape();
    ion_ = s.stepType().ionization();
    sp_ = s.simParticle().isNonnull() ? int(s.simParticle().key()) : -1;
    eion_ = s.ionizingEdep();
    len_ = s.stepLength();
    width_ = s.width();
    t_ = s.time();
    x0_ = s.startPosition().x();
    y0_ = s.startPosition().y();
    z0_ = s.startPosition().z();
    x1_ = s.endPosition().x();
    y1_ = s.endPosition().y();
    z1_ = s.endPosition().z();
    px_ = s.momentum().x();
    py_ = s.momentum().y();
    pz_ = s.momentum().z();
    sgs_->Fill();
  }

  art::Handle<SurfaceStepCollection> ssh;
  if (event.getByLabel(conf_.surfaceSteps(), ssh)) {
    for (auto const& s : *ssh) {
      surf_ = s.surfaceId().id().id();
      sindex_ = s.surfaceId().index();
      ssp_ = s.simParticle().isNonnull() ? int(s.simParticle().key()) : -1;
      sedep_ = s.energyDeposit();
      st_ = s.time();
      sx0_ = s.startPosition().x();
      sy0_ = s.startPosition().y();
      sz0_ = s.startPosition().z();
      sx1_ = s.endPosition().x();
      sy1_ = s.endPosition().y();
      sz1_ = s.endPosition().z();
      spx_ = s.momentum().x();
      spy_ = s.momentum().y();
      spz_ = s.momentum().z();
      ss_->Fill();
    }
  }

  art::Handle<StepPointMCCollection> vh;
  if (event.getByLabel(conf_.vdSteps(), vh)) {
    for (auto const& s : *vh) {
      vdid_ = s.volumeId();
      vsp_ = s.simParticle().isNonnull() ? int(s.simParticle().key()) : -1;
      vx_ = s.position().x();
      vy_ = s.position().y();
      vz_ = s.position().z();
      vpx_ = s.momentum().x();
      vpy_ = s.momentum().y();
      vpz_ = s.momentum().z();
      vt_ = s.time();
      vd_->Fill();
    }
  }

  if (traj_) {
    art::Handle<MCTrajectoryCollection> th;
    if (event.getByLabel(conf_.mcTrajectories(), th)) {
      for (auto const& [simp, traj] : *th) {
        tsp_ = simp.isNonnull() ? int(simp.key()) : -1;
        tpt_ = 0;
        for (auto const& pt : traj.points()) {
          tx_ = pt.x();
          ty_ = pt.y();
          tz_ = pt.z();
          tt_ = pt.t();
          tke_ = pt.kineticEnergy();
          traj_->Fill();
          ++tpt_;
        }
      }
    }
  }

  std::vector<size_t> mcprim;
  art::Handle<PrimaryParticle> pph;
  if (event.getByLabel(conf_.primaryParticle(), pph))
    for (auto const& p : pph->primarySimParticles()) mcprim.push_back(p.key());
  auto ph = event.getValidHandle<SimParticleCollection>(conf_.simParticles());
  for (auto const& [k, p] : *ph) {
    key_ = int(k.asUint());
    parent_ = p.parent().isNonnull() ? int(p.parent().key()) : -1;
    pdg_ = p.pdgId();
    ccode_ = p.creationCode().id();
    scode_ = p.stoppingCode().id();
    isprim_ = p.isPrimary();
    ismcprim_ = std::find(mcprim.begin(), mcprim.end(), size_t(k.asUint())) != mcprim.end();
    genidx_ = p.genParticle().isNonnull() ? int(p.genParticle().key()) : -1;
    psx_ = p.startPosition().x();
    psy_ = p.startPosition().y();
    psz_ = p.startPosition().z();
    pex_ = p.endPosition().x();
    pey_ = p.endPosition().y();
    pez_ = p.endPosition().z();
    pspx_ = p.startMomentum().px();
    pspy_ = p.startMomentum().py();
    pspz_ = p.startMomentum().pz();
    pse_ = p.startMomentum().e();
    pepx_ = p.endMomentum().px();
    pepy_ = p.endMomentum().py();
    pepz_ = p.endMomentum().pz();
    pee_ = p.endMomentum().e();
    pst_ = p.startGlobalTime();
    pet_ = p.endGlobalTime();
    parts_->Fill();
  }
}

void StrawGasStepTableDumper::endSubRun(const art::SubRun& subrun) {
  run_ = subrun.run();
  subrun_ = subrun.subRun();
  art::Handle<CosmicLivetime> lh;
  subrun.getByLabel(conf_.livetime(), lh);
  livetime_ = lh.isValid() ? lh->liveTime() : -1;
  primaries_ = lh.isValid() ? lh->primaries() : 0;
  subruns_->Fill();
}

}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::StrawGasStepTableDumper)
