// Lossless detector-only table and optional in-event comparison of original and replayed collections.
#include <cstdint>
#include <string>
#include <vector>

#include "TTree.h"
#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art_root_io/TFileService.h"
#include "canvas/Utilities/InputTag.h"
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"
#include "Offline/RecoDataProducts/inc/StrawDigi.hh"

namespace mu2e {
class StrawDigiTableDumper : public art::EDAnalyzer {
 public:
  struct Config {
    fhicl::Atom<art::InputTag> digis{fhicl::Name("StrawDigiTag"), fhicl::Comment("digis and aligned ADC waveforms"), art::InputTag("makeSD")};
    fhicl::Atom<art::InputTag> compare{fhicl::Name("compareTo"), fhicl::Comment("optional exact reference collection"), art::InputTag("")};
  };
  using Parameters = art::EDAnalyzer::Table<Config>;
  explicit StrawDigiTableDumper(Parameters const& config)
      : art::EDAnalyzer(config), tag_(config().digis()), compare_(config().compare()) {}
  void beginJob() override;
  void analyze(art::Event const& event) override;
 private:
  art::InputTag tag_, compare_;
  TTree* tree_{nullptr};
  unsigned run_, subrun_, event_;
  std::vector<unsigned> sid_, tdc0_, tdc1_, tot0_, tot1_, pmp_, flags_, nadc_, adc_;
};

void StrawDigiTableDumper::beginJob() {
  art::ServiceHandle<art::TFileService> tfs;
  tree_ = tfs->make<TTree>("events", "lossless StrawDigi and ADC collections");
  tree_->Branch("run", &run_);
  tree_->Branch("subrun", &subrun_);
  tree_->Branch("event", &event_);
  for (auto [name, values] : std::vector<std::pair<const char*, std::vector<unsigned>*>>{
           {"d_sid", &sid_}, {"d_tdc0", &tdc0_}, {"d_tdc1", &tdc1_}, {"d_tot0", &tot0_}, {"d_tot1", &tot1_},
           {"d_pmp", &pmp_}, {"d_flags", &flags_}, {"d_nadc", &nadc_}, {"d_adc", &adc_}})
    tree_->Branch(name, values);
}

void StrawDigiTableDumper::analyze(art::Event const& event) {
  auto const& digis = *event.getValidHandle<StrawDigiCollection>(tag_);
  auto const& waveforms = *event.getValidHandle<StrawDigiADCWaveformCollection>(tag_);
  if (digis.size() != waveforms.size()) throw cet::exception("TRKDIGITABLE") << "unaligned ADC collection in " << event.id();
  if (!compare_.label().empty()) {
    auto const& reference = *event.getValidHandle<StrawDigiCollection>(compare_);
    auto const& refADC = *event.getValidHandle<StrawDigiADCWaveformCollection>(compare_);
    if (reference.size() != digis.size() || refADC.size() != waveforms.size())
      throw cet::exception("TRKDIGITABLE") << "round-trip collection count mismatch in " << event.id();
    for (std::size_t i = 0; i < digis.size(); ++i) {
      auto const& a = reference[i];
      auto const& b = digis[i];
      if (a.strawId().asUint16() != b.strawId().asUint16() || a.TDC() != b.TDC() || a.TOT() != b.TOT() ||
          a.PMP() != b.PMP() || a.digiFlag().hex() != b.digiFlag().hex() || refADC[i].samples() != waveforms[i].samples())
        throw cet::exception("TRKDIGITABLE") << "round-trip mismatch in " << event.id() << " digi " << i;
    }
  }
  run_ = event.run(); subrun_ = event.subRun(); event_ = event.event();
  for (auto* values : {&sid_, &tdc0_, &tdc1_, &tot0_, &tot1_, &pmp_, &flags_, &nadc_, &adc_}) values->clear();
  for (std::size_t i = 0; i < digis.size(); ++i) {
    auto const& digi = digis[i];
    sid_.push_back(digi.strawId().asUint16());
    tdc0_.push_back(digi.TDC()[0]); tdc1_.push_back(digi.TDC()[1]);
    tot0_.push_back(digi.TOT()[0]); tot1_.push_back(digi.TOT()[1]);
    pmp_.push_back(digi.PMP());
    flags_.push_back(static_cast<unsigned>(std::stoul(digi.digiFlag().hex(), nullptr, 16)));
    auto const& adc = waveforms[i].samples();
    nadc_.push_back(adc.size());
    adc_.insert(adc_.end(), adc.begin(), adc.end());
  }
  tree_->Fill();
}
}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::StrawDigiTableDumper)
