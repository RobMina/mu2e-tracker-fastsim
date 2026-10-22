// Dump StrawDigis, stock makeSH/makePH ComboHits and the StrawResponse constants to flat TTrees (T-VR1 reference).
#include <cmath>
#include <cstdint>
#include <string>
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
#include "Offline/RecoDataProducts/inc/CosmicTrackSeed.hh"
#include "Offline/RecoDataProducts/inc/KalSeed.hh"
#include "Offline/RecoDataProducts/inc/TimeCluster.hh"
#include "Offline/DataProducts/inc/StrawIdMask.hh"
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
    fhicl::Atom<bool> dumpTracks{Name("DumpTracks"), Comment("also dump time clusters, line seeds, KalSeed fits"), false};
    fhicl::Atom<art::InputTag> tc{Name("TimeClusterTag"), Comment("time clusters"), "SimpleTimeCluster"};
    fhicl::Atom<art::InputTag> seed{Name("CosmicTrackSeedTag"), Comment("line seeds"), "LineFinder"};
    fhicl::Atom<art::InputTag> kal{Name("KalSeedTag"), Comment("line fits"), "KKLine"};
  };
  using Parameters = art::EDAnalyzer::Table<Config>;
  explicit StrawHitTableDumper(Parameters const& c);
  void beginJob() override;
  void analyze(art::Event const& e) override;

 private:
  void dumpConditions(art::Event const& e);
  void fillCH(ComboHitCollection const& col, bool panel);
  void fillTracks(art::Event const& e);
  art::InputTag sdTag_, shTag_, phTag_, pbtTag_, ewmTag_, tcTag_, seedTag_, kalTag_;
  bool dumpTracks_;
  // time clusters: hit counts, t0, flat makeSH indices
  std::vector<int> tcn_, tcind_;
  std::vector<float> tct0_, tct0err_;
  // line seeds: time-cluster index, status word, t0 and the line (A0, 0, B0) + s (A1, -1, B1), flat makeSH indices
  std::vector<int> sdtc_, sdn_, sdind_;
  std::vector<unsigned> sdstat_;
  std::vector<double> sdt0_, sda0_, sdb0_, sda1_, sdb1_;
  // KalSeed fits: status, chi2, ndof, consistency, t0, KinematicLine at the segment nearest t0, hit counts
  std::vector<unsigned> klstat_;
  std::vector<int> klndof_, klnhit_, klnact_;
  std::vector<float> klchi2_, klcon_, klt0_, klt0err_;
  std::vector<double> klpar_[6], klerr_[6];
  // KalSeed hits: track index, makeSH index, strawId, ambiguity, active, fit DOCA, drift radius, fit dt
  std::vector<int> khtrk_, khind_, khsid_, khamb_, khact_;
  std::vector<float> khdoca_, khrdrift_, khdt_, khrdt_, khsderr_, khrdoca_;
  std::vector<float> khsignq_, khdriftq_, khcdrift_, khudocavar_, khwdot_;   // drift ANN outputs and inputs
  // KalSeed straws (material crossings): track index, strawId, active, TOCA, wall/gas/wire path lengths, X0 fraction
  std::vector<int> ksttrk_, kstsid_, kstact_;
  std::vector<float> ksttoca_, kstwall_, kstgas_, kstwire_, kstradlen_;
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
  std::vector<double> dtgrid_, rdrift_, dvel_, sderr_, uderr_;   // driftInfo(dt, phi=0) on a dt grid (KKLine drift residual)
  std::vector<double> wcalx_, wcaly_, wcalz_, whvx_, whvy_, whvz_;  // wire ends (cal, hv) per straw: KKLine signal lines
  double tdcLSB_, adcLSB_, totLSB_, gain_, eionq_, pmpavg_, rout_;
  int npre_;
};

StrawHitTableDumper::StrawHitTableDumper(Parameters const& c)
    : art::EDAnalyzer{c}, sdTag_(c().sd()), shTag_(c().sh()), phTag_(c().ph()), pbtTag_(c().pbt()), ewmTag_(c().ewm()),
      tcTag_(c().tc()), seedTag_(c().seed()), kalTag_(c().kal()), dumpTracks_(c().dumpTracks()) {}

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
  if (dumpTracks_) {
    for (auto [n, v] : std::vector<std::pair<const char*, std::vector<int>*>>{
             {"tc_n", &tcn_}, {"tc_ind", &tcind_}, {"sd_tc", &sdtc_}, {"sd_n", &sdn_}, {"sd_ind", &sdind_},
             {"kl_ndof", &klndof_}, {"kl_nhit", &klnhit_}, {"kl_nactive", &klnact_}, {"kh_trk", &khtrk_},
             {"kh_ind", &khind_}, {"kh_sid", &khsid_}, {"kh_ambig", &khamb_}, {"kh_active", &khact_}})
      ev_->Branch(n, v);
    ev_->Branch("sd_status", &sdstat_);
    ev_->Branch("kl_status", &klstat_);
    for (auto [n, v] : std::vector<std::pair<const char*, std::vector<float>*>>{
             {"tc_t0", &tct0_}, {"tc_t0err", &tct0err_}, {"kl_chi2", &klchi2_}, {"kl_fitcon", &klcon_},
             {"kl_t0", &klt0_}, {"kl_t0err", &klt0err_}, {"kh_doca", &khdoca_}, {"kh_rdrift", &khrdrift_},
             {"kh_dt", &khdt_}, {"kh_rdt", &khrdt_}, {"kh_sderr", &khsderr_}, {"kh_rdoca", &khrdoca_},
             {"kh_signqual", &khsignq_}, {"kh_driftqual", &khdriftq_}, {"kh_cdrift", &khcdrift_},
             {"kh_udocavar", &khudocavar_}, {"kh_wdot", &khwdot_},
             {"kst_toca", &ksttoca_}, {"kst_wall", &kstwall_}, {"kst_gas", &kstgas_}, {"kst_wire", &kstwire_},
             {"kst_radlen", &kstradlen_}})
      ev_->Branch(n, v);
    for (auto [n, v] : std::vector<std::pair<const char*, std::vector<int>*>>{
             {"kst_trk", &ksttrk_}, {"kst_sid", &kstsid_}, {"kst_active", &kstact_}})
      ev_->Branch(n, v);
    for (auto [n, v] : std::vector<std::pair<const char*, std::vector<double>*>>{
             {"sd_t0", &sdt0_}, {"sd_a0", &sda0_}, {"sd_b0", &sdb0_}, {"sd_a1", &sda1_}, {"sd_b1", &sdb1_}})
      ev_->Branch(n, v);
    const char* pn[6] = {"d0", "phi0", "z0", "theta", "t0", "mom"};  // KinKal::KinematicLine::ParamIndex order
    for (int i = 0; i < 6; ++i) {
      ev_->Branch((std::string("kl_") + pn[i]).c_str(), &klpar_[i]);
      ev_->Branch((std::string("kl_") + pn[i] + "err").c_str(), &klerr_[i]);
    }
  }
  cond_ = tfs->make<TTree>("conditions", "StrawResponse/StrawElectronics/geometry used by makeSH (first event)");
  for (auto [n, v] : std::vector<std::pair<const char*, std::vector<int>*>>{
           {"sid", &csid_}, {"dead", &cdead_}, {"noisy", &cnoisy_}})
    cond_->Branch(n, v);
  for (auto [n, v] : std::vector<std::pair<const char*, std::vector<double>*>>{
           {"midx", &mx_}, {"midy", &my_}, {"midz", &mz_}, {"dirx", &dx_}, {"diry", &dy_}, {"dirz", &dz_},
           {"hlen", &hlen_}, {"pmpscale", &pmps_}, {"toff_hv", &toffhv_}, {"toff_cal", &toffcal_}, {"hpv2", &hpv2_},
           {"kgrid", &kgrid_}, {"hpv_s0", &hpvs0_}, {"wpres0", &wpres0_}, {"wpres200", &wpres200_},
           {"totdt", &totdt_}, {"totde", &totde_}, {"dtgrid", &dtgrid_}, {"rdrift", &rdrift_}, {"dvel", &dvel_},
           {"sderr", &sderr_}, {"uderr", &uderr_}, {"wcalx", &wcalx_}, {"wcaly", &wcaly_}, {"wcalz", &wcalz_}, {"whvx", &whvx_},
           {"whvy", &whvy_}, {"whvz", &whvz_}})
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
    auto const wc = straw.wireEnd(StrawEnd::cal);
    auto const wh = straw.wireEnd(StrawEnd::hv);
    wcalx_.push_back(wc.x()); wcaly_.push_back(wc.y()); wcalz_.push_back(wc.z());
    whvx_.push_back(wh.x()); whvy_.push_back(wh.y()); whvz_.push_back(wh.z());
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
  for (int i = 0; i <= 2000; ++i) {  // dt -20 .. 80 ns in 0.05 ns steps
    double dt = -20.0 + 0.05 * i;
    auto di = srep.driftInfo(s0.id(), dt, 0.0);
    dtgrid_.push_back(dt); rdrift_.push_back(di.rDrift_); dvel_.push_back(di.driftVelocity_);
    sderr_.push_back(di.signedDriftError_); uderr_.push_back(di.unsignedDriftError_);
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
    c.flag.push_back(uint32_t(std::stoul(ch.flag().hex(), nullptr, 16)));  // BitMap exposes its word only as hex()
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

void StrawHitTableDumper::fillTracks(art::Event const& e) {
  for (auto* v : {&tcn_, &tcind_, &sdtc_, &sdn_, &sdind_, &klndof_, &klnhit_, &klnact_, &khtrk_, &khind_, &khsid_,
                  &khamb_, &khact_}) v->clear();
  for (auto* v : {&sdstat_, &klstat_}) v->clear();
  for (auto* v : {&ksttrk_, &kstsid_, &kstact_}) v->clear();
  for (auto* v : {&ksttoca_, &kstwall_, &kstgas_, &kstwire_, &kstradlen_, &khsignq_, &khdriftq_, &khcdrift_,
                  &khudocavar_, &khwdot_}) v->clear();
  for (auto* v : {&tct0_, &tct0err_, &klchi2_, &klcon_, &klt0_, &klt0err_, &khdoca_, &khrdrift_, &khdt_, &khrdt_, &khsderr_,
                  &khrdoca_}) v->clear();
  for (auto* v : {&sdt0_, &sda0_, &sdb0_, &sda1_, &sdb1_}) v->clear();
  for (int i = 0; i < 6; ++i) { klpar_[i].clear(); klerr_[i].clear(); }
  auto const& tcH = e.getValidHandle<TimeClusterCollection>(tcTag_);
  for (auto const& tc : *tcH) {
    tcn_.push_back(tc.nhits());
    for (auto i : tc.hits()) tcind_.push_back(i);
    tct0_.push_back(tc.t0().t0()); tct0err_.push_back(tc.t0().t0Err());
  }
  for (auto const& sd : *e.getValidHandle<CosmicTrackSeedCollection>(seedTag_)) {
    sdtc_.push_back(sd.timeCluster().isNonnull() ? int(sd.timeCluster().key()) : -1);
    sdstat_.push_back(uint32_t(std::stoul(sd.status().hex(), nullptr, 16)));
    auto const& fp = sd.track().FitParams;
    sdt0_.push_back(fp.T0); sda0_.push_back(fp.A0); sdb0_.push_back(fp.B0); sda1_.push_back(fp.A1); sdb1_.push_back(fp.B1);
    std::vector<StrawHitIndex> idx;
    if (!sd.hits().empty()) sd.hits().fillStrawHitIndices(idx, StrawIdMask::uniquestraw);
    sdn_.push_back(idx.size());
    sdind_.insert(sdind_.end(), idx.begin(), idx.end());
  }
  int itrk = 0;
  for (auto const& ks : *e.getValidHandle<KalSeedCollection>(kalTag_)) {
    klstat_.push_back(uint32_t(std::stoul(ks.status().hex(), nullptr, 16)));
    klchi2_.push_back(ks.chisquared()); klndof_.push_back(ks.nDOF()); klcon_.push_back(ks.fitConsistency());
    klt0_.push_back(ks.t0Val()); klt0err_.push_back(std::sqrt(ks.t0Var()));
    auto seg = ks.segments().empty() ? ks.segments().end() : ks.nearestSegment(ks.t0Val());
    for (int i = 0; i < 6; ++i) {
      if (seg == ks.segments().end()) { klpar_[i].push_back(0); klerr_[i].push_back(-1); continue; }
      auto kl = seg->kinematicLine();
      klpar_[i].push_back(kl.paramVal(i)); klerr_[i].push_back(std::sqrt(kl.paramVar(i)));
    }
    int nact = 0;
    for (auto const& h : ks.hits()) {
      bool act = h.flag().hasAllProperties(StrawHitFlag::active);
      nact += act;
      khtrk_.push_back(itrk); khind_.push_back(h.index()); khsid_.push_back(h.strawId().asUint16());
      khamb_.push_back(h.ambig()); khact_.push_back(act);
      khdoca_.push_back(h.fitDOCA()); khrdrift_.push_back(h.driftRadius()); khdt_.push_back(h.fitDt());
      khrdt_.push_back(h.refDt()); khsderr_.push_back(h.radialErr()); khrdoca_.push_back(h.refDOCA());
      khsignq_.push_back(h._signqual); khdriftq_.push_back(h._driftqual); khcdrift_.push_back(h._cdrift);
      khudocavar_.push_back(h._udocavar); khwdot_.push_back(h._wdot);
    }
    klnhit_.push_back(ks.hits().size()); klnact_.push_back(nact);
    for (auto const& st : ks.straws()) {
      ksttrk_.push_back(itrk); kstsid_.push_back(st._straw.asUint16()); kstact_.push_back(st.active());
      ksttoca_.push_back(st._toca); kstwall_.push_back(st._wallpath); kstgas_.push_back(st._gaspath);
      kstwire_.push_back(st._wirepath); kstradlen_.push_back(st._radlen);
    }
    ++itrk;
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
  if (dumpTracks_) fillTracks(e);
  ev_->Fill();
}

}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::StrawHitTableDumper)
