// Dump StrawDigis, stock makeSH/makePH ComboHits and the StrawResponse constants to flat TTrees (T-VR1 reference).
#include <cstdint>
#include <vector>

#include "TTree.h"
#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Principal/Event.h"
#include "art_root_io/TFileService.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/types/Atom.h"

#include "Offline/DataProducts/inc/EventWindowMarker.hh"
#include "Offline/DataProducts/inc/StrawEnd.hh"
#include "Offline/DataProducts/inc/StrawId.hh"
#include "Offline/ProditionsService/inc/ProditionsHandle.hh"
#include "Offline/RecoDataProducts/inc/ComboHit.hh"
#include "Offline/RecoDataProducts/inc/ProtonBunchTime.hh"
#include "Offline/RecoDataProducts/inc/StrawDigi.hh"
#include "Offline/TrackerConditions/inc/StrawElectronics.hh"
#include "Offline/TrackerConditions/inc/StrawResponse.hh"
#include "Offline/TrackerConditions/inc/TrackerStatus.hh"
#include "Offline/TrackerGeom/inc/Tracker.hh"

namespace mu2e {

class StrawHitTableDumper : public art::EDAnalyzer {
 public:
  struct Config {
    using Name = fhicl::Name;
    using Comment = fhicl::Comment;
    fhicl::Atom<art::InputTag> sd{Name("StrawDigiTag"), Comment("StrawDigis + ADC waveforms"), "makeSD"};
    fhicl::Atom<art::InputTag> sh{Name("StrawHitTag"), Comment("single-straw ComboHits"), "makeSH"};
    fhicl::Atom<art::InputTag> ph{Name("PanelHitTag"), Comment("panel ComboHits"), "makePH"};
    fhicl::Atom<art::InputTag> pbt{Name("ProtonBunchTimeTag"), Comment("PBT used by makeSH"), "PBTFSD"};
    fhicl::Atom<art::InputTag> ewm{Name("EventWindowMarkerTag"), Comment("EWM"), "EWMProducer"};
  };
  using Parameters = art::EDAnalyzer::Table<Config>;
  explicit StrawHitTableDumper(Parameters const& c);
  void beginJob() override;
  void analyze(art::Event const& e) override;

 private:
  void dumpConditions(art::Event const& e);
  void fillCH(ComboHitCollection const& col, bool panel);
  art::InputTag sdTag_, shTag_, phTag_, pbtTag_, ewmTag_;
  ProditionsHandle<StrawResponse> srep_h_;
  ProditionsHandle<StrawElectronics> sele_h_;
  ProditionsHandle<TrackerStatus> tstat_h_;
  ProditionsHandle<Tracker> trk_h_;
  TTree *ev_{nullptr}, *cond_{nullptr};
  bool condDone_{false};
  // event row
  unsigned run_, subrun_, event_;
  float pbt_;
  int spill_;
  std::vector<int> dsid_, dtdc0_, dtdc1_, dtot0_, dtot1_, dpmp_, dnadc_, dadc_;
  // single-straw (s_) and panel (p_) hits share one layout; p_ adds the constituent indices
  struct CH {
    std::vector<int> sid, nsh, ncombo, eend, ind;
    std::vector<unsigned> flag;
    std::vector<float> time, et0, et1, tot0, tot1, edep, dtime, ptime, wdist, uvar, vvar, timevar, qual, x, y, z;
  } s_, p_;
  // conditions (one row)
  std::vector<int> csid_, cdead_, cnoisy_;
  std::vector<double> mx_, my_, mz_, dx_, dy_, dz_, hlen_, pmps_, toffhv_, toffcal_, hpv2_;
  std::vector<double> kgrid_, hpvs0_, wpres0_, wpres200_, totdt_, totde_;
  double tdcLSB_, adcLSB_, totLSB_, gain_, eionq_, pmpavg_, rout_;
  int npre_;
};

StrawHitTableDumper::StrawHitTableDumper(Parameters const& c)
    : art::EDAnalyzer{c}, sdTag_(c().sd()), shTag_(c().sh()), phTag_(c().ph()), pbtTag_(c().pbt()), ewmTag_(c().ewm()) {}

void StrawHitTableDumper::beginJob() {
  art::ServiceHandle<art::TFileService> tfs;
  ev_ = tfs->make<TTree>("events", "digis + stock makeSH/makePH ComboHits per event");
  ev_->Branch("run", &run_);
  ev_->Branch("subrun", &subrun_);
  ev_->Branch("event", &event_);
  ev_->Branch("pbt", &pbt_);
  ev_->Branch("spill", &spill_);
  for (auto [n, v] : std::vector<std::pair<const char*, std::vector<int>*>>{
           {"d_sid", &dsid_}, {"d_tdc0", &dtdc0_}, {"d_tdc1", &dtdc1_}, {"d_tot0", &dtot0_}, {"d_tot1", &dtot1_},
           {"d_pmp", &dpmp_}, {"d_nadc", &dnadc_}, {"d_adc", &dadc_}})
    ev_->Branch(n, v);
  for (auto [pre, ch] : std::vector<std::pair<std::string, CH*>>{{"s_", &s_}, {"p_", &p_}}) {
    for (auto [n, v] : std::vector<std::pair<const char*, std::vector<int>*>>{
             {"sid", &ch->sid}, {"nsh", &ch->nsh}, {"ncombo", &ch->ncombo}, {"eend", &ch->eend}, {"ind", &ch->ind}})
      ev_->Branch((pre + n).c_str(), v);
    ev_->Branch((pre + "flag").c_str(), &ch->flag);
    for (auto [n, v] : std::vector<std::pair<const char*, std::vector<float>*>>{
             {"time", &ch->time}, {"et0", &ch->et0}, {"et1", &ch->et1}, {"tot0", &ch->tot0}, {"tot1", &ch->tot1},
             {"edep", &ch->edep}, {"dtime", &ch->dtime}, {"ptime", &ch->ptime}, {"wdist", &ch->wdist},
             {"uvar", &ch->uvar}, {"vvar", &ch->vvar}, {"timevar", &ch->timevar}, {"qual", &ch->qual},
             {"x", &ch->x}, {"y", &ch->y}, {"z", &ch->z}})
      ev_->Branch((pre + n).c_str(), v);
  }
  cond_ = tfs->make<TTree>("conditions", "StrawResponse/StrawElectronics/geometry used by makeSH (first event)");
  for (auto [n, v] : std::vector<std::pair<const char*, std::vector<int>*>>{
           {"sid", &csid_}, {"dead", &cdead_}, {"noisy", &cnoisy_}})
    cond_->Branch(n, v);
  for (auto [n, v] : std::vector<std::pair<const char*, std::vector<double>*>>{
           {"midx", &mx_}, {"midy", &my_}, {"midz", &mz_}, {"dirx", &dx_}, {"diry", &dy_}, {"dirz", &dz_},
           {"hlen", &hlen_}, {"pmpscale", &pmps_}, {"toff_hv", &toffhv_}, {"toff_cal", &toffcal_}, {"hpv2", &hpv2_},
           {"kgrid", &kgrid_}, {"hpv_s0", &hpvs0_}, {"wpres0", &wpres0_}, {"wpres200", &wpres200_},
           {"totdt", &totdt_}, {"totde", &totde_}})
    cond_->Branch(n, v);
  for (auto [n, v] : std::vector<std::pair<const char*, double*>>{
           {"tdcLSB", &tdcLSB_}, {"adcLSB", &adcLSB_}, {"totLSB", &totLSB_}, {"strawGain", &gain_},
           {"eionPerQ", &eionq_}, {"pmpScaleAvg", &pmpavg_}, {"strawOuterRadius", &rout_}})
    cond_->Branch(n, v);
  cond_->Branch("nADCPreSamples", &npre_);
}

void StrawHitTableDumper::dumpConditions(art::Event const& e) {
  auto const& srep = srep_h_.get(e.id());
  auto const& sele = sele_h_.get(e.id());
  auto const& ts = tstat_h_.get(e.id());
  auto const& tt = trk_h_.get(e.id());
  tdcLSB_ = sele.tdcLSB(); adcLSB_ = srep.adcLSB(); totLSB_ = srep.totLSB(); npre_ = srep.nADCPreSamples();
  gain_ = srep.strawGain(); eionq_ = srep.ionizationEnergy(1.0); pmpavg_ = srep.peakMinusPedestalEnergyScale();
  rout_ = tt.strawOuterRadius();
  TrkTypes::TDCValues tdc0{0, 0};
  for (auto const& straw : tt.getStraws()) {
    StrawId sid = straw.id();
    TrkTypes::TDCTimes t;
    srep.calibrateTimes(tdc0, t, sid);
    csid_.push_back(sid.asUint16());
    cdead_.push_back(ts.noSignal(sid) || ts.suppress(sid));
    cnoisy_.push_back(ts.noisy(sid));
    auto const& m = straw.getMidPoint();
    auto const& d = straw.getDirection();
    mx_.push_back(m.x()); my_.push_back(m.y()); mz_.push_back(m.z());
    dx_.push_back(d.x()); dy_.push_back(d.y()); dz_.push_back(d.z());
    hlen_.push_back(straw.halfLength());
    pmps_.push_back(srep.peakMinusPedestalEnergyScale(sid));
    toffhv_.push_back(t[StrawEnd::hv]); toffcal_.push_back(t[StrawEnd::cal]);
    hpv2_.push_back(srep.halfPropV(sid, 2.0));
  }
  auto const& s0 = tt.getStraws().front();
  for (int i = 0; i <= 1200; ++i) {
    double k = 0.005 * i;
    kgrid_.push_back(k);
    hpvs0_.push_back(srep.halfPropV(s0.id(), k));
    wpres0_.push_back(srep.wpRes(k, 0.0));
    wpres200_.push_back(srep.wpRes(k, 200.0));
  }
  for (int it = 0; it < 100; ++it)  // TOT 0..199 ns in 2 ns steps x edep 0..0.0045 MeV in 0.0005 steps (bin probe)
    for (int ie = 0; ie < 10; ++ie) {
      totdt_.push_back(srep.TOTdriftTime(s0, 2.0 * it + 1.0, 0.0005 * ie + 0.00025));
      totde_.push_back(srep.TOTdriftTimeError(s0, 2.0 * it + 1.0, 0.0005 * ie + 0.00025));
    }
  cond_->Fill();
  condDone_ = true;
}

void StrawHitTableDumper::fillCH(ComboHitCollection const& col, bool panel) {
  CH& c = panel ? p_ : s_;
  for (auto const& ch : col) {
    c.sid.push_back(ch.strawId().asUint16());
    c.nsh.push_back(ch.nStrawHits());
    c.ncombo.push_back(ch.nCombo());
    c.eend.push_back(ch.earlyEnd().end());
    for (size_t i = 0; i < ch.nCombo(); ++i) c.ind.push_back(ch.index(i));
    c.flag.push_back(ch.flag().hash());
    c.time.push_back(ch.correctedTime());
    c.et0.push_back(ch.endTime(StrawEnd::cal));
    c.et1.push_back(ch.endTime(StrawEnd::hv));
    c.tot0.push_back(ch.TOTs()[StrawEnd::cal]);
    c.tot1.push_back(ch.TOTs()[StrawEnd::hv]);
    c.edep.push_back(ch.energyDep());
    c.dtime.push_back(ch.driftTime());
    c.ptime.push_back(ch.propTime());
    c.wdist.push_back(ch.wireDist());
    c.uvar.push_back(ch.uVar());
    c.vvar.push_back(ch.vVar());
    c.timevar.push_back(ch.timeVar());
    c.qual.push_back(ch.qual());
    c.x.push_back(ch.pos().x()); c.y.push_back(ch.pos().y()); c.z.push_back(ch.pos().z());
  }
}

void StrawHitTableDumper::analyze(art::Event const& e) {
  if (!condDone_) dumpConditions(e);
  run_ = e.run(); subrun_ = e.subRun(); event_ = e.event();
  pbt_ = e.getValidHandle<ProtonBunchTime>(pbtTag_)->pbtime_;
  spill_ = static_cast<int>(e.getValidHandle<EventWindowMarker>(ewmTag_)->spillType());
  for (auto* v : {&dsid_, &dtdc0_, &dtdc1_, &dtot0_, &dtot1_, &dpmp_, &dnadc_, &dadc_}) v->clear();
  for (CH* c : {&s_, &p_}) {
    for (auto* v : {&c->sid, &c->nsh, &c->ncombo, &c->eend, &c->ind}) v->clear();
    c->flag.clear();
    for (auto* v : {&c->time, &c->et0, &c->et1, &c->tot0, &c->tot1, &c->edep, &c->dtime, &c->ptime, &c->wdist,
                    &c->uvar, &c->vvar, &c->timevar, &c->qual, &c->x, &c->y, &c->z})
      v->clear();
  }
  auto const& sd = *e.getValidHandle<StrawDigiCollection>(sdTag_);
  auto const& wf = *e.getValidHandle<StrawDigiADCWaveformCollection>(sdTag_);
  for (size_t i = 0; i < sd.size(); ++i) {
    auto const& d = sd[i];
    dsid_.push_back(d.strawId().asUint16());
    dtdc0_.push_back(d.TDC()[StrawEnd::cal]); dtdc1_.push_back(d.TDC()[StrawEnd::hv]);
    dtot0_.push_back(d.TOT()[StrawEnd::cal]); dtot1_.push_back(d.TOT()[StrawEnd::hv]);
    dpmp_.push_back(d.PMP());
    auto const& s = wf.at(i).samples();
    dnadc_.push_back(s.size());
    dadc_.insert(dadc_.end(), s.begin(), s.end());
  }
  fillCH(*e.getValidHandle<ComboHitCollection>(shTag_), false);
  fillCH(*e.getValidHandle<ComboHitCollection>(phTag_), true);
  ev_->Fill();
}

}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::StrawHitTableDumper)
