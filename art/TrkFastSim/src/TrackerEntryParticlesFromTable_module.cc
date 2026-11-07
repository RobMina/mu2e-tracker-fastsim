// Replay tracker-entry muons in table index order on the original RootInput event stream.
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>

#include "TBranch.h"
#include "TFile.h"
#include "TLeaf.h"
#include "TObjArray.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art_root_io/TFileService.h"
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"
#include "Offline/GlobalConstantsService/inc/GlobalConstantsHandle.hh"
#include "Offline/GlobalConstantsService/inc/ParticleDataList.hh"
#include "Offline/MCDataProducts/inc/GenParticle.hh"

namespace mu2e {
namespace {
using Key = std::tuple<UInt_t, UInt_t, UInt_t>;
struct Particle {
  Long64_t source_sp{};
  Int_t pdg{};
  Double_t x{}, y{}, z{}, px{}, py{}, pz{}, t{};
};
struct EntryEvent {
  UInt_t count;
  std::map<UInt_t, Particle> particles;
};

void scalar(TTree* tree, const char* name, const char* type) {
  auto* branch = tree->GetBranch(name);
  auto* leaf = branch ? branch->GetLeaf(name) : nullptr;
  if (!leaf || std::string(branch->GetClassName()) != "" || branch->GetListOfLeaves()->GetEntries() != 1 ||
      std::string(leaf->GetTypeName()) != type || leaf->GetLeafCount() || leaf->GetLenStatic() != 1 ||
      std::string(leaf->GetTitle()).find('[') != std::string::npos || branch->GetEntries() != tree->GetEntries())
    throw cet::exception("TRKENTRYTABLE") << tree->GetName() << "/" << name << " must be a scalar " << type
                                          << " with one value per tree row";
}

template <class T> T value(TTreeReaderValue<T>& column) {
  if (column.GetSetupStatus() < 0 || !column.Get())
    throw cet::exception("TRKENTRYTABLE") << "unreadable column " << column.GetBranchName();
  return *column;
}

struct Keys {
  TTreeReaderValue<UInt_t> run, subrun, event;
  explicit Keys(TTreeReader& reader) : run(reader, "run"), subrun(reader, "subrun"), event(reader, "event") {}
  Key get() {
    const auto r = value(run), s = value(subrun), e = value(event);
    const auto invalid = std::numeric_limits<UInt_t>::max();
    if (r == 0 || e == 0 || r == invalid || s == invalid || e == invalid)
      throw cet::exception("TRKENTRYTABLE") << "invalid art event key " << r << "/" << s << "/" << e;
    return Key(r, s, e);
  }
};

template <class F> void scan(TTree* tree, TTreeReader& reader, F consume) {
  const auto expected = tree->GetEntries();
  if (expected < 0) throw cet::exception("TRKENTRYTABLE") << "invalid entry count in " << tree->GetName();
  // Empty TTrees have already had their schema checked and need no reader setup.
  if (expected == 0) return;
  Long64_t seen = 0;
  while (reader.Next()) { consume(); ++seen; }
  if (seen != expected || reader.GetEntryStatus() != TTreeReader::kEntryBeyondEnd)
    throw cet::exception("TRKENTRYTABLE") << "incomplete read of " << tree->GetName() << ": " << seen << "/"
                                          << expected << ", reader status " << reader.GetEntryStatus();
}

double magnitude(Particle const& p) { return std::hypot(std::hypot(p.px, p.py), p.pz); }
}  // namespace

class TrackerEntryParticlesFromTable : public art::EDProducer {
 public:
  struct Config {
    fhicl::Atom<std::string> tableFile{fhicl::Name("tableFile"), fhicl::Comment("tracker-entry ROOT TTree file")};
    fhicl::Atom<std::string> directory{fhicl::Name("directory"), fhicl::Comment("input TTree directory"), "entry"};
    fhicl::Atom<bool> writeReadback{fhicl::Name("writeReadback"), fhicl::Comment("write replayed states via TFileService"), false};
  };
  using Parameters = art::EDProducer::Table<Config>;
  explicit TrackerEntryParticlesFromTable(Parameters const& config);
  void beginJob() override;
  void produce(art::Event& event) override;
 private:
  void readback(art::Event const& event, GenParticleCollection const& output, EntryEvent const& input);
  std::map<Key, EntryEvent> entries_;
  bool writeReadback_;
  TTree* eventsOut_{nullptr};
  TTree* particlesOut_{nullptr};
  UInt_t run_{}, subrun_{}, event_{}, count_{}, index_{};
  Particle replay_{};
  Double_t energy_{}, properTime_{};
  Int_t genId_{};
};

TrackerEntryParticlesFromTable::TrackerEntryParticlesFromTable(Parameters const& config)
    : art::EDProducer(config), writeReadback_(config().writeReadback()) {
  produces<GenParticleCollection>();
  if (config().tableFile().empty() || config().directory().empty())
    throw cet::exception("TRKENTRYTABLE") << "tableFile and directory must be nonempty";
  std::unique_ptr<TFile> file(TFile::Open(config().tableFile().c_str(), "READ"));
  if (!file || file->IsZombie()) throw cet::exception("TRKENTRYTABLE") << "cannot open " << config().tableFile();
  auto* events = file->Get<TTree>((config().directory() + "/events").c_str());
  auto* particles = file->Get<TTree>((config().directory() + "/particles").c_str());
  if (!events || !particles) throw cet::exception("TRKENTRYTABLE") << "missing events or particles TTree";
  for (auto* tree : {events, particles})
    for (auto name : {"run", "subrun", "event"}) scalar(tree, name, "UInt_t");
  scalar(events, "n_particles", "UInt_t");
  scalar(particles, "index", "UInt_t");
  scalar(particles, "source_sp", "Long64_t");
  scalar(particles, "pdg", "Int_t");
  for (auto name : {"x", "y", "z", "px", "py", "pz", "t"}) scalar(particles, name, "Double_t");
  {
    TTreeReader reader(events);
    Keys keys(reader);
    TTreeReaderValue<UInt_t> count(reader, "n_particles");
    scan(events, reader, [&] {
      if (!entries_.emplace(keys.get(), EntryEvent{value(count), {}}).second)
        throw cet::exception("TRKENTRYTABLE") << "duplicate event key";
    });
  }
  {
    TTreeReader reader(particles);
    Keys keys(reader);
    TTreeReaderValue<UInt_t> index(reader, "index");
    TTreeReaderValue<Long64_t> source(reader, "source_sp");
    TTreeReaderValue<Int_t> pdg(reader, "pdg");
    TTreeReaderValue<Double_t> x(reader, "x"), y(reader, "y"), z(reader, "z"), t(reader, "t");
    TTreeReaderValue<Double_t> px(reader, "px"), py(reader, "py"), pz(reader, "pz");
    scan(particles, reader, [&] {
      auto found = entries_.find(keys.get());
      if (found == entries_.end()) throw cet::exception("TRKENTRYTABLE") << "particle without an events row";
      const UInt_t i = value(index);
      if (i >= found->second.count) throw cet::exception("TRKENTRYTABLE") << "index outside declared count: " << i;
      Particle p{value(source), value(pdg), value(x), value(y), value(z), value(px), value(py), value(pz), value(t)};
      if (p.pdg != 13 && p.pdg != -13)
        throw cet::exception("TRKENTRYTABLE") << "unsupported PDG " << p.pdg << "; only +/-13 are supported";
      for (double v : {p.x, p.y, p.z, p.px, p.py, p.pz, p.t})
        if (!std::isfinite(v)) throw cet::exception("TRKENTRYTABLE") << "nonfinite state for source_sp " << p.source_sp;
      if (!(magnitude(p) > 0.) || !std::isfinite(magnitude(p)))
        throw cet::exception("TRKENTRYTABLE") << "invalid momentum for source_sp " << p.source_sp;
      if (!found->second.particles.emplace(i, p).second)
        throw cet::exception("TRKENTRYTABLE") << "duplicate particle index " << i;
    });
  }
  for (auto const& item : entries_) {
    auto const& entry = item.second;
    if (entry.particles.size() != entry.count)
      throw cet::exception("TRKENTRYTABLE") << "particle count mismatch in " << std::get<0>(item.first) << "/"
                                            << std::get<1>(item.first) << "/" << std::get<2>(item.first);
    UInt_t next = 0;
    std::set<Long64_t> sources;
    for (auto const& particle : entry.particles) {
      if (particle.first != next++) throw cet::exception("TRKENTRYTABLE") << "noncontiguous particle indices";
      if (!sources.insert(particle.second.source_sp).second)
        throw cet::exception("TRKENTRYTABLE") << "duplicate source_sp in event: " << particle.second.source_sp;
    }
  }
}

void TrackerEntryParticlesFromTable::beginJob() {
  if (!writeReadback_) return;
  art::ServiceHandle<art::TFileService> tfs;
  eventsOut_ = tfs->make<TTree>("events", "replayed event counts including empty collections");
  particlesOut_ = tfs->make<TTree>("particles", "actual GenParticle states and input source mapping");
  for (auto* tree : {eventsOut_, particlesOut_}) {
    tree->Branch("run", &run_, "run/i");
    tree->Branch("subrun", &subrun_, "subrun/i");
    tree->Branch("event", &event_, "event/i");
  }
  eventsOut_->Branch("n_particles", &count_, "n_particles/i");
  particlesOut_->Branch("index", &index_, "index/i");
  particlesOut_->Branch("source_sp", &replay_.source_sp, "source_sp/L");
  particlesOut_->Branch("pdg", &replay_.pdg, "pdg/I");
  particlesOut_->Branch("x", &replay_.x, "x/D");
  particlesOut_->Branch("y", &replay_.y, "y/D");
  particlesOut_->Branch("z", &replay_.z, "z/D");
  particlesOut_->Branch("px", &replay_.px, "px/D");
  particlesOut_->Branch("py", &replay_.py, "py/D");
  particlesOut_->Branch("pz", &replay_.pz, "pz/D");
  particlesOut_->Branch("t", &replay_.t, "t/D");
  particlesOut_->Branch("energy", &energy_, "energy/D");
  particlesOut_->Branch("proper_time", &properTime_, "proper_time/D");
  particlesOut_->Branch("gen_id", &genId_, "gen_id/I");
}

void TrackerEntryParticlesFromTable::produce(art::Event& event) {
  auto found = entries_.find(Key(event.run(), event.subRun(), event.event()));
  if (found == entries_.end()) throw cet::exception("TRKENTRYTABLE") << "event absent from table: " << event.id();
  auto output = std::make_unique<GenParticleCollection>();
  output->reserve(found->second.count);
  for (auto const& item : found->second.particles) {
    auto const& p = item.second;
    const double mass = GlobalConstantsHandle<ParticleDataList>()->particle(p.pdg).mass();
    const double energy = std::hypot(magnitude(p), mass);
    if (!std::isfinite(mass) || mass <= 0. || !std::isfinite(energy))
      throw cet::exception("TRKENTRYTABLE") << "invalid mass or energy for PDG " << p.pdg;
    output->emplace_back(PDGCode::type(p.pdg), GenId::fromStepPointMCs, CLHEP::Hep3Vector(p.x, p.y, p.z),
                         CLHEP::HepLorentzVector(p.px, p.py, p.pz, energy), p.t);
  }
  if (writeReadback_) readback(event, *output, found->second);
  event.put(std::move(output));
}

void TrackerEntryParticlesFromTable::readback(art::Event const& event, GenParticleCollection const& output,
                                             EntryEvent const& input) {
  run_ = event.run(); subrun_ = event.subRun(); event_ = event.event();
  count_ = static_cast<UInt_t>(output.size());
  if (eventsOut_->Fill() < 0) throw cet::exception("TRKENTRYTABLE") << "failed to write readback events";
  for (index_ = 0; index_ < count_; ++index_) {
    auto const& gen = output.at(index_);
    // Only source_sp comes from the input row; all particle states come from the output object.
    replay_ = Particle{input.particles.at(index_).source_sp, static_cast<Int_t>(gen.pdgId()),
                       gen.position().x(), gen.position().y(), gen.position().z(),
                       gen.momentum().px(), gen.momentum().py(), gen.momentum().pz(), gen.time()};
    energy_ = gen.momentum().e(); properTime_ = gen.properTime(); genId_ = static_cast<Int_t>(gen.generatorId().id());
    if (particlesOut_->Fill() < 0) throw cet::exception("TRKENTRYTABLE") << "failed to write readback particles";
  }
}
}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::TrackerEntryParticlesFromTable)
