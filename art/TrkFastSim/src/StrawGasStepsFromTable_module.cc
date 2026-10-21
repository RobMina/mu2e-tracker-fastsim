// Path A: fast-sim StrawGasStep (+ optional CrvStep) tables -> the products compressDetStepMCs writes, for stock digi/reco.
// Run with module label compressDetStepMCs on the gen stream of the same CRY primaries; events matched by (run, subrun, event).
// One SimParticle tree for both detectors: the CRV table is read here too, so CrvStepsFromTable is not used in this chain.
// Tables: dump/sgsteps + dump/particles (trkfastsim.steps), optional dump/surfsteps, dump/trajpts; CRV dump/steps + dump/particles.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "TFile.h"
#include "TLeaf.h"
#include "TTree.h"
#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "canvas/Persistency/Common/Ptr.h"
#include "canvas/Utilities/InputTag.h"
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"
#include "messagefacility/MessageLogger/MessageLogger.h"

#include "Offline/DataProducts/inc/PDGCode.hh"
#include "Offline/DataProducts/inc/StrawId.hh"
#include "Offline/DataProducts/inc/SurfaceId.hh"
#include "Offline/GeometryService/inc/DetectorSystem.hh"
#include "Offline/GeometryService/inc/GeomHandle.hh"
#include "Offline/MCDataProducts/inc/CaloShowerStep.hh"
#include "Offline/MCDataProducts/inc/CrvStep.hh"
#include "Offline/MCDataProducts/inc/GenParticle.hh"
#include "Offline/MCDataProducts/inc/MCTrajectory.hh"
#include "Offline/MCDataProducts/inc/ProcessCode.hh"
#include "Offline/MCDataProducts/inc/SimParticle.hh"
#include "Offline/MCDataProducts/inc/StepPointMC.hh"
#include "Offline/MCDataProducts/inc/StrawGasStep.hh"
#include "Offline/MCDataProducts/inc/SurfaceStep.hh"

namespace mu2e {

namespace {
using Key = std::tuple<unsigned, unsigned, unsigned>;

// every leaf of a flat TTree, read as double whatever numpy dtype uproot wrote
struct Leaves {
  TTree* t{nullptr};
  std::map<std::string, TLeaf*> m;
  explicit Leaves(TTree* tree) : t(tree) {
    TIter next(t->GetListOfLeaves());
    while (auto* l = static_cast<TLeaf*>(next())) m[l->GetName()] = l;
  }
  bool has(const std::string& n) const { return m.count(n) != 0; }
  double operator()(const std::string& n) const {
    auto it = m.find(n);
    if (it == m.end()) throw cet::exception("TRKFASTSIM") << "column " << n << " missing in tree " << t->GetName();
    return it->second->GetValue(0);
  }
  double get(const std::string& n, double def) const {
    auto it = m.find(n);
    return it == m.end() ? def : it->second->GetValue(0);
  }
  Key key() const { return Key(unsigned((*this)("run")), unsigned((*this)("subrun")), unsigned((*this)("event"))); }
};

template <class F> void loop(TTree* t, F f) {
  Leaves L(t);
  for (Long64_t i = 0; i < t->GetEntries(); ++i) {
    t->GetEntry(i);
    f(L);
  }
}

double massOf(int pdg) {
  switch (std::abs(pdg)) {
    case 11: return 0.51099895;
    case 13: return 105.6583755;
    case 211: return 139.57039;
    case 321: return 493.677;
    case 2212: return 938.27208816;
    case 2112: return 939.56542052;
    default: return 0.;
  }
}
}  // namespace

class StrawGasStepsFromTable : public art::EDProducer {
 public:
  struct Config {
    fhicl::Atom<std::string> trkTableFile{fhicl::Name("trkTableFile"), fhicl::Comment("trkfastsim table (dump/sgsteps, dump/particles)")};
    fhicl::Atom<std::string> crvTableFile{fhicl::Name("crvTableFile"), fhicl::Comment("crvfastsim table (dump/steps, dump/particles); empty = no CrvSteps"), ""};
    fhicl::Atom<art::InputTag> genParticles{fhicl::Name("genParticles"), fhicl::Comment("GenParticles the primaries link to (gen stream: every event)"), art::InputTag("cosmicTimeOffset")};
    fhicl::Atom<long> trkKeyOffset{fhicl::Name("trkKeyOffset"), fhicl::Comment("added to tracker SimParticle keys other than primaryKey (0 = keys as in the table)"), 0};
    fhicl::Atom<long> primaryKey{fhicl::Name("primaryKey"), fhicl::Comment("key of the primary muon, shared by both tables"), 1};
    fhicl::Atom<bool> makeTrajectories{fhicl::Name("makeTrajectories"), fhicl::Comment("without a trajpts tree, synthesise MCTrajectories of primaries from their steps"), true};
  };
  using Parameters = art::EDProducer::Table<Config>;
  explicit StrawGasStepsFromTable(const Parameters& conf);
  void produce(art::Event& event) override;

 private:
  struct Sgs { int plane, panel, straw, shape, ion; long sp; float eion, len, width, x0, y0, z0, x1, y1, z1, px, py, pz; double t; };
  struct Crv { int bar; long sp; float edep, x0, y0, z0, x1, y1, z1, px0, py0, pz0, p1, len; double t0, t1; };
  struct Surf { int surf, sindex; long sp; float edep, x0, y0, z0, x1, y1, z1, px, py, pz; double t; };
  struct Traj { long sp; int ipt; float x, y, z, t, ke; };
  struct Part {
    long parent{-1};
    int pdg{0};
    bool prim{false};
    long genidx{-1};
    bool hasStart{false}, hasEnd{false};
    double sx{0}, sy{0}, sz{0}, ex{0}, ey{0}, ez{0};
    float spx{0}, spy{0}, spz{0}, st{0}, epx{0}, epy{0}, epz{0}, et{0};
  };
  void readParticles(TTree* t, bool trk);
  long remap(long k) const { return (k < 0 || k == primKey_ || trkOff_ == 0) ? k : k + trkOff_; }

  art::InputTag genTag_;
  long trkOff_, primKey_;
  bool makeTraj_;
  std::map<Key, std::vector<Sgs>> sgs_;
  std::map<Key, std::vector<Crv>> crv_;
  std::map<Key, std::vector<Surf>> surf_;
  std::map<Key, std::vector<Traj>> traj_;
  std::map<Key, std::map<long, Part>> parts_;
  bool haveSurf_{false}, haveTraj_{false};
};

void StrawGasStepsFromTable::readParticles(TTree* t, bool trk) {
  loop(t, [&](const Leaves& L) {
    long k = long(L("key"));
    Part p;
    p.parent = long(L("parent"));
    if (trk) { k = remap(k); p.parent = remap(p.parent); }
    p.pdg = int(L("pdg"));
    p.prim = L("isprim") != 0;
    p.genidx = long(L.get("genidx", -1));
    if (L.has("sx")) {
      p.hasStart = p.hasEnd = true;
      p.sx = L("sx"); p.sy = L("sy"); p.sz = L("sz"); p.ex = L("ex"); p.ey = L("ey"); p.ez = L("ez");
      p.spx = L("spx"); p.spy = L("spy"); p.spz = L("spz"); p.st = L("st");
      p.epx = L("epx"); p.epy = L("epy"); p.epz = L("epz"); p.et = L("et");
    }
    auto& ev = parts_[L.key()];
    auto it = ev.find(k);
    if (it == ev.end()) { ev.emplace(k, p); return; }
    if (it->second.pdg != p.pdg || it->second.parent != p.parent)
      throw cet::exception("TRKFASTSIM") << "SimParticle key " << k << " used by different particles in the tracker and CRV tables"
                                          << " (run/subrun/event " << std::get<0>(L.key()) << "/" << std::get<1>(L.key()) << "/"
                                          << std::get<2>(L.key()) << "); set trkKeyOffset";
    if (!it->second.hasStart && p.hasStart) it->second = p;
  });
}

StrawGasStepsFromTable::StrawGasStepsFromTable(const Parameters& conf)
    : art::EDProducer(conf), genTag_(conf().genParticles()), trkOff_(conf().trkKeyOffset()), primKey_(conf().primaryKey()),
      makeTraj_(conf().makeTrajectories()) {
  produces<SimParticleCollection>();
  produces<StrawGasStepCollection>();
  produces<CrvStepCollection>();
  produces<MCTrajectoryCollection>();
  produces<SurfaceStepCollection>();
  produces<StepPointMCCollection>("virtualdetector");
  produces<CaloShowerStepCollection>();

  std::unique_ptr<TFile> f(TFile::Open(conf().trkTableFile().c_str()));
  if (!f || f->IsZombie()) throw cet::exception("TRKFASTSIM") << "cannot open " << conf().trkTableFile();
  auto* dir = f->GetDirectory("dump");
  auto* pt = dir ? dir->Get<TTree>("particles") : nullptr;
  auto* st = dir ? dir->Get<TTree>("sgsteps") : nullptr;
  if (!pt || !st) throw cet::exception("TRKFASTSIM") << "no dump/particles or dump/sgsteps TTree in " << conf().trkTableFile();
  readParticles(pt, true);
  loop(st, [&](const Leaves& L) {
    sgs_[L.key()].push_back(Sgs{int(L("plane")), int(L("panel")), int(L("straw")), int(L("shape")), int(L("ion")), remap(long(L("sp"))),
                                float(L("eion")), float(L("len")), float(L("width")), float(L("x0")), float(L("y0")), float(L("z0")),
                                float(L("x1")), float(L("y1")), float(L("z1")), float(L("px")), float(L("py")), float(L("pz")), L("t")});
  });
  if (auto* ss = dir->Get<TTree>("surfsteps")) {
    haveSurf_ = true;
    loop(ss, [&](const Leaves& L) {
      surf_[L.key()].push_back(Surf{int(L("surf")), int(L.get("sindex", 0)), remap(long(L("sp"))), float(L.get("edep", 0)),
                                    float(L("x0")), float(L("y0")), float(L("z0")), float(L("x1")), float(L("y1")), float(L("z1")),
                                    float(L("px")), float(L("py")), float(L("pz")), L("t")});
    });
  } else {
    mf::LogWarning("TRKFASTSIM") << "no dump/surfsteps in " << conf().trkTableFile()
                                 << ": SurfaceStepCollection will be empty (no trksegsmc truth)";
  }
  if (auto* tt = dir->Get<TTree>("trajpts")) {
    haveTraj_ = true;
    loop(tt, [&](const Leaves& L) {
      traj_[L.key()].push_back(Traj{remap(long(L("sp"))), int(L("ipt")), float(L("x")), float(L("y")), float(L("z")), float(L("t")), float(L("ke"))});
    });
  } else {
    mf::LogWarning("TRKFASTSIM") << "no dump/trajpts in " << conf().trkTableFile()
                                 << (makeTraj_ ? ": MCTrajectories synthesised for primaries from generation point and steps"
                                               : ": MCTrajectoryCollection will be empty");
  }

  if (!conf().crvTableFile().empty()) {
    std::unique_ptr<TFile> fc(TFile::Open(conf().crvTableFile().c_str()));
    if (!fc || fc->IsZombie()) throw cet::exception("TRKFASTSIM") << "cannot open " << conf().crvTableFile();
    auto* dc = fc->GetDirectory("dump");
    auto* cp = dc ? dc->Get<TTree>("particles") : nullptr;
    auto* cs = dc ? dc->Get<TTree>("steps") : nullptr;
    if (!cp || !cs) throw cet::exception("TRKFASTSIM") << "no dump/particles or dump/steps TTree in " << conf().crvTableFile();
    readParticles(cp, false);
    loop(cs, [&](const Leaves& L) {
      crv_[L.key()].push_back(Crv{int(L("bar")), long(L("sp")), float(L("edep")), float(L("x0")), float(L("y0")), float(L("z0")),
                                  float(L("x1")), float(L("y1")), float(L("z1")), float(L("px0")), float(L("py0")), float(L("pz0")),
                                  float(L("p1")), float(L("len")), L("t0"), L("t1")});
    });
  } else {
    mf::LogInfo("TRKFASTSIM") << "no crvTableFile: CrvStepCollection will be empty";
  }
  mf::LogInfo("TRKFASTSIM") << "read " << sgs_.size() << " events with StrawGasSteps, " << crv_.size() << " with CrvSteps";
}

void StrawGasStepsFromTable::produce(art::Event& event) {
  auto sims = std::make_unique<SimParticleCollection>();
  auto sgsOut = std::make_unique<StrawGasStepCollection>();
  auto crvOut = std::make_unique<CrvStepCollection>();
  auto trajOut = std::make_unique<MCTrajectoryCollection>();
  auto surfOut = std::make_unique<SurfaceStepCollection>();
  Key key(event.run(), event.subRun(), event.event());
  auto ip = parts_.find(key);
  if (ip != parts_.end()) {
    GeomHandle<DetectorSystem> det;
    auto pid = event.getProductID<SimParticleCollection>();
    auto* getter = event.productGetter(pid);
    auto ptr = [&](long k) { return art::Ptr<SimParticle>(pid, std::size_t(k), getter); };
    auto const& parts = ip->second;
    auto known = [&](long k) { return parts.count(k) != 0; };
    static const std::vector<Sgs> noSgs;
    static const std::vector<Crv> noCrv;
    auto is = sgs_.find(key);
    auto ic = crv_.find(key);
    auto const& sgs = is == sgs_.end() ? noSgs : is->second;
    auto const& crv = ic == crv_.end() ? noCrv : ic->second;

    // per particle: time-ordered points (Mu2e frame) with |p|, from the steps
    struct Pt { double t; CLHEP::Hep3Vector x; CLHEP::Hep3Vector p; };
    std::map<long, std::vector<Pt>> pts;
    for (auto const& s : sgs) {
      CLHEP::Hep3Vector p(s.px, s.py, s.pz);
      double beta = 1.;
      double m = known(s.sp) ? massOf(parts.at(s.sp).pdg) : 0.;
      if (p.mag() > 0) beta = p.mag() / std::sqrt(p.mag2() + m * m);
      pts[s.sp].push_back(Pt{s.t, det->toMu2e(CLHEP::Hep3Vector(s.x0, s.y0, s.z0)), p});
      pts[s.sp].push_back(Pt{s.t + s.len / (beta * 299.792458), det->toMu2e(CLHEP::Hep3Vector(s.x1, s.y1, s.z1)), p});
    }
    for (auto const& c : crv) {
      CLHEP::Hep3Vector p(c.px0, c.py0, c.pz0);
      pts[c.sp].push_back(Pt{c.t0, CLHEP::Hep3Vector(c.x0, c.y0, c.z0), p});
      pts[c.sp].push_back(Pt{c.t1, CLHEP::Hep3Vector(c.x1, c.y1, c.z1), p.mag() > 0 ? p * (c.p1 / p.mag()) : p});
    }
    for (auto& [k, v] : pts) std::sort(v.begin(), v.end(), [](Pt const& a, Pt const& b) { return a.t < b.t; });

    art::Handle<GenParticleCollection> gh;
    event.getByLabel(genTag_, gh);
    std::vector<bool> genUsed(gh.isValid() ? gh->size() : 0, false);
    auto lv = [](CLHEP::Hep3Vector const& p, double m) { return CLHEP::HepLorentzVector(p, std::sqrt(p.mag2() + m * m)); };

    for (auto const& [k, p] : parts) {
      cet::map_vector_key mk(static_cast<unsigned long>(k));
      double m = massOf(p.pdg);
      art::Ptr<GenParticle> gp;
      if (p.prim && gh.isValid()) {
        long gi = p.genidx;
        if (gi < 0)
          for (std::size_t i = 0; i < gh->size(); ++i)
            if (!genUsed[i] && gh->at(i).pdgId() == p.pdg) { gi = long(i); break; }
        if (gi >= 0 && std::size_t(gi) < gh->size()) { gp = art::Ptr<GenParticle>(gh, std::size_t(gi)); genUsed[gi] = true; }
      }
      auto ipt = pts.find(k);
      bool hasPts = ipt != pts.end() && !ipt->second.empty();
      CLHEP::Hep3Vector x0, p0, x1, p1;
      double t0 = 0, t1 = 0;
      if (p.hasStart) {
        x0 = CLHEP::Hep3Vector(p.sx, p.sy, p.sz); p0 = CLHEP::Hep3Vector(p.spx, p.spy, p.spz); t0 = p.st;
        x1 = CLHEP::Hep3Vector(p.ex, p.ey, p.ez); p1 = CLHEP::Hep3Vector(p.epx, p.epy, p.epz); t1 = p.et;
      } else {
        if (gp.isNonnull()) {
          x0 = gp->position(); p0 = gp->momentum().vect(); t0 = gp->time();
        } else if (hasPts) {
          x0 = ipt->second.front().x; p0 = ipt->second.front().p; t0 = ipt->second.front().t;
        }
        if (hasPts) { x1 = ipt->second.back().x; p1 = ipt->second.back().p; t1 = ipt->second.back().t; }
        else { x1 = x0; p1 = p0; t1 = t0; }
      }
      art::Ptr<SimParticle> par;
      if (p.parent >= 0 && known(p.parent)) par = ptr(p.parent);
      ProcessCode code(p.prim ? ProcessCode::mu2ePrimary : ProcessCode::unknown);
      (*sims)[mk] = SimParticle(mk, 0, par, PDGCode::type(p.pdg), gp, x0, lv(p0, m), float(t0), 0.f, 0u, 0u, code);
      auto e1 = lv(p1, m);
      (*sims)[mk].addEndInfo(x1, e1, float(t1), 0.f, 0u, 0u, ProcessCode(ProcessCode::unknown), float(e1.e() - m), 0, 0.f);
    }
    for (auto const& [k, p] : parts)
      if (p.parent >= 0 && known(p.parent)) (*sims)[cet::map_vector_key(static_cast<unsigned long>(p.parent))].addDaughter(ptr(k));

    for (auto const& s : sgs) {
      if (!known(s.sp)) throw cet::exception("TRKFASTSIM") << "StrawGasStep with unknown sp " << s.sp << " in " << event.id();
      StrawGasStep::StepType type(StrawGasStep::StepType::Shape(s.shape), StrawGasStep::StepType::Ionization(s.ion));
      sgsOut->emplace_back(StrawId(uint16_t(s.plane), uint16_t(s.panel), uint16_t(s.straw)), type, s.eion, s.len, s.width, s.t,
                           XYZVectorF(s.x0, s.y0, s.z0), XYZVectorF(s.x1, s.y1, s.z1), XYZVectorF(s.px, s.py, s.pz), ptr(s.sp));
    }
    for (auto const& c : crv) {
      if (!known(c.sp)) throw cet::exception("TRKFASTSIM") << "CrvStep with unknown sp " << c.sp << " in " << event.id();
      crvOut->emplace_back(CRSScintillatorBarIndex(c.bar), c.edep, c.t0, c.t1, XYZVectorF(c.x0, c.y0, c.z0), XYZVectorF(c.x1, c.y1, c.z1),
                           XYZVectorF(c.px0, c.py0, c.pz0), c.p1, c.len, ptr(c.sp));
    }
    auto isu = surf_.find(key);
    if (isu != surf_.end())
      for (auto const& s : isu->second) {
        if (!known(s.sp)) continue;
        CLHEP::Hep3Vector a = det->toMu2e(CLHEP::Hep3Vector(s.x0, s.y0, s.z0)), b = det->toMu2e(CLHEP::Hep3Vector(s.x1, s.y1, s.z1));
        CLHEP::Hep3Vector mom(s.px, s.py, s.pz);
        StepPointMC spmc(ptr(s.sp), 0, s.edep, 0., s.edep, s.t, 0., a, b, mom, mom, (b - a).mag(), ProcessCode(ProcessCode::unknown));
        surfOut->emplace_back(SurfaceId(SurfaceIdDetail::enum_type(s.surf), s.sindex), spmc, det);
      }
    auto itr = traj_.find(key);
    if (itr != traj_.end()) {
      std::map<long, std::vector<Traj>> byp;
      for (auto const& t : itr->second) byp[t.sp].push_back(t);
      for (auto& [k, v] : byp) {
        if (!known(k)) continue;
        std::sort(v.begin(), v.end(), [](Traj const& a, Traj const& b) { return a.ipt < b.ipt; });
        std::vector<MCTrajectoryPoint> tp;
        for (auto const& t : v) tp.emplace_back(CLHEP::Hep3Vector(t.x, t.y, t.z), t.t, t.ke);
        (*trajOut)[ptr(k)] = MCTrajectory(ptr(k), tp);
      }
    } else if (makeTraj_ && !haveTraj_) {
      // straight-line primaries: generation point, then each step entry/exit in time order
      for (auto const& [k, p] : parts) {
        if (!p.prim) continue;
        auto const& sp = (*sims)[cet::map_vector_key(static_cast<unsigned long>(k))];
        double m = massOf(p.pdg);
        std::vector<MCTrajectoryPoint> tp;
        tp.emplace_back(sp.startPosition(), sp.startGlobalTime(), float(sp.startMomentum().e() - m));
        auto ipt = pts.find(k);
        if (ipt != pts.end())
          for (auto const& q : ipt->second) tp.emplace_back(q.x, float(q.t), float(std::sqrt(q.p.mag2() + m * m) - m));
        (*trajOut)[ptr(k)] = MCTrajectory(ptr(k), tp);
      }
    }
  } else if (sgs_.count(key) || crv_.count(key)) {
    throw cet::exception("TRKFASTSIM") << "steps without particles in event " << event.id();
  }
  event.put(std::move(sims));
  event.put(std::move(sgsOut));
  event.put(std::move(crvOut));
  event.put(std::move(trajOut));
  event.put(std::move(surfOut));
  event.put(std::make_unique<StepPointMCCollection>(), "virtualdetector");
  event.put(std::make_unique<CaloShowerStepCollection>());
}

}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::StrawGasStepsFromTable)
