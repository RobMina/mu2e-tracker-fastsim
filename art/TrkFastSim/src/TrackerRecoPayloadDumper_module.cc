// Export complete ROOT-streamed persistent reco payloads for strict same-configuration replay checks.
#include <string>
#include <vector>

#include "TBufferFile.h"
#include "TClass.h"
#include "TTree.h"
#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art_root_io/TFileService.h"
#include "canvas/Utilities/InputTag.h"
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"
#include "Offline/RecoDataProducts/inc/BkgCluster.hh"
#include "Offline/RecoDataProducts/inc/BkgClusterHit.hh"
#include "Offline/RecoDataProducts/inc/ComboHit.hh"
#include "Offline/RecoDataProducts/inc/CosmicTrackSeed.hh"
#include "Offline/RecoDataProducts/inc/IntensityInfoTimeCluster.hh"
#include "Offline/RecoDataProducts/inc/IntensityInfoTrackerHits.hh"
#include "Offline/RecoDataProducts/inc/KalSeed.hh"
#include "Offline/RecoDataProducts/inc/KalSeedAssns.hh"
#include "Offline/RecoDataProducts/inc/ProtonBunchTime.hh"
#include "Offline/RecoDataProducts/inc/StrawHit.hh"
#include "Offline/RecoDataProducts/inc/TimeCluster.hh"

namespace mu2e {
class TrackerRecoPayloadDumper : public art::EDAnalyzer {
 public:
  struct Config {
    fhicl::Atom<std::string> flagPHModuleType{fhicl::Name("flagPHModuleType"), fhicl::Comment("expanded flagPH module type"), "FlagBkgHits"};
    fhicl::Atom<art::InputTag> usedPBT{fhicl::Name("usedPBT"), fhicl::Comment("PBT consumed by makeSH"), art::InputTag("PBTFSD")};
    fhicl::Atom<bool> backgrounds{fhicl::Name("backgrounds"), fhicl::Comment("include saved FlagBkgHits background collections"), true};
  };
  using Parameters = art::EDAnalyzer::Table<Config>;
  explicit TrackerRecoPayloadDumper(Parameters const& config)
      : art::EDAnalyzer(config), flagPHType_(config().flagPHModuleType()), usedPBT_(config().usedPBT()), backgrounds_(config().backgrounds()) {
    if (flagPHType_ != "NullProducer" && flagPHType_ != "FlagBkgHits")
      throw cet::exception("TRKDIGITABLE") << "unsupported flagPH module type " << flagPHType_;
  }
  void beginJob() override;
  void analyze(art::Event const& event) override;
 private:
  template <class T> void stream(art::Event const& event, const char* tag, std::vector<unsigned char>& bytes) {
    auto const& product = *event.getValidHandle<T>(art::InputTag(tag));
    auto* cls = TClass::GetClass(typeid(T));
    if (!cls || !cls->GetStreamerInfo()) throw cet::exception("TRKDIGITABLE") << "missing payload dictionary for " << tag;
    TBufferFile buffer(TBuffer::kWrite);
    buffer.WriteObjectAny(&product, cls);
    auto const* begin = reinterpret_cast<const unsigned char*>(buffer.Buffer());
    bytes.assign(begin, begin + buffer.Length());
  }
  std::string flagPHType_;
  art::InputTag usedPBT_;
  bool backgrounds_;
  unsigned char flagPHActive_{}, backgroundsActive_{};
  TTree* tree_{nullptr};
  unsigned run_, subrun_, event_;
  std::vector<unsigned char> pbt_, usedPBTBytes_, sh_, ph_, sth_, flagged_, strawHits_, hitIntensity_, tcIntensity_, tc_, seeds_, kal_, assns_, bkg_, bkgHits_;
};

void TrackerRecoPayloadDumper::beginJob() {
  art::ServiceHandle<art::TFileService> tfs;
  tree_ = tfs->make<TTree>("events", "complete persistent reco payloads; identical process/module labels required");
  tree_->Branch("run", &run_);
  tree_->Branch("subrun", &subrun_);
  tree_->Branch("event", &event_);
  flagPHActive_ = flagPHType_ == "FlagBkgHits";
  backgroundsActive_ = flagPHActive_ && backgrounds_;
  tree_->Branch("flag_ph_active", &flagPHActive_);
  tree_->Branch("backgrounds_active", &backgroundsActive_);
  for (auto [name, bytes] : std::vector<std::pair<const char*, std::vector<unsigned char>*>>{
           {"pbt", &pbt_}, {"used_pbt", &usedPBTBytes_}, {"single_hits", &sh_}, {"panel_hits", &ph_}, {"stereo_hits", &sth_},
           {"flagged_hits", &flagged_}, {"straw_hits", &strawHits_}, {"hit_intensity", &hitIntensity_},
           {"cluster_intensity", &tcIntensity_}, {"time_clusters", &tc_}, {"line_seeds", &seeds_},
           {"kal_seeds", &kal_}, {"kal_line_assns", &assns_}})
    tree_->Branch(name, bytes);
  tree_->Branch("background_clusters", &bkg_);
  tree_->Branch("background_hits", &bkgHits_);
}

void TrackerRecoPayloadDumper::analyze(art::Event const& event) {
  run_ = event.run(); subrun_ = event.subRun(); event_ = event.event();
  stream<ProtonBunchTime>(event, "PBTFSD", pbt_);
  stream<ProtonBunchTime>(event, usedPBT_.encode().c_str(), usedPBTBytes_);
  stream<ComboHitCollection>(event, "makeSH", sh_);
  stream<ComboHitCollection>(event, "makePH", ph_);
  stream<ComboHitCollection>(event, "makeSTH", sth_);
  if (flagPHActive_) {
    stream<ComboHitCollection>(event, "flagPH", flagged_);
    stream<IntensityInfoTimeCluster>(event, "flagPH", tcIntensity_);
  }
  stream<StrawHitCollection>(event, "makeSH", strawHits_);
  stream<IntensityInfoTrackerHits>(event, "makeSH", hitIntensity_);
  stream<TimeClusterCollection>(event, "SimpleTimeCluster", tc_);
  stream<CosmicTrackSeedCollection>(event, "LineFinder", seeds_);
  stream<KalSeedCollection>(event, "KKLine", kal_);
  stream<KalLineAssns>(event, "KKLine", assns_);
  if (backgroundsActive_) {
    stream<BkgClusterCollection>(event, "flagPH", bkg_);
    stream<BkgClusterHitCollection>(event, "flagPH", bkgHits_);
  }
  tree_->Fill();
}
}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::TrackerRecoPayloadDumper)
