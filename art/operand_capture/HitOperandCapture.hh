#ifndef TrkFastSim_HitOperandCapture_hh
#define TrkFastSim_HitOperandCapture_hh
#include "KinKal/Fit/IterationTrace.hh"
#include "KinKal/Detector/Hit.hh"
#include "Offline/Mu2eKinKal/inc/KKStrawHit.hh"
#include "art_root_io/TFileService.h"
#include "TTree.h"
#include <limits>
#include <stdexcept>
#include <vector>
#include <optional>
#include <utility>

namespace mu2e {
  template<class KTRAJ> void captureHitOperands(KinKal::Hit<KTRAJ> const& hit, std::optional<KinKal::HitOperandTrace>& payload) {
    auto const* sh = dynamic_cast<KKStrawHit<KTRAJ> const*>(&hit);
    if(!sh)return;
    KinKal::HitOperandTrace out;
    out.shindex = sh->strawHitIndex();
    out.strawid = sh->strawId().asUint16();
    auto const& ca = sh->closestApproach();
    auto const& state = ca.tpData();
    out.caStatus = state.status(); out.caUsable = state.usable();
    out.precision = ca.precision();
    out.hitState = sh->hitState().state_;
    out.constrainTOT = sh->hitState().constrainTOT();
    out.constrainLong = sh->hitState().constrainLong();
    out.constrainAbsDrift = sh->hitState().constrainAbsDriftDt();
    if(out.caUsable){
      auto const& p = state.particlePoca(); auto const& s = state.sensorPoca();
      auto const& u = state.particleDirection(); auto const& w = state.sensorDirection();
      out.particlePoca = {p.X(),p.Y(),p.Z(),p.T()}; out.sensorPoca = {s.X(),s.Y(),s.Z(),s.T()};
      out.particleDirection = {u.X(),u.Y(),u.Z()}; out.sensorDirection = {w.X(),w.Y(),w.Z()};
      out.doca = state.doca(); out.docaVariance = state.docaVar();
      out.tocaVariance = state.tocaVar(); out.lsign = state.lSign();
      for(size_t j=0; j<KinKal::NParams(); ++j){
        out.dDdP.push_back(ca.dDdP()[j]); out.dTdP.push_back(ca.dTdP()[j]);
      }
    }
    auto const& rows = sh->refResiduals();
    for(auto const& row : rows){
      out.rowActive.push_back(row.active() ? 1 : 0);
      out.rowValue.push_back(row.value()); out.rowMvar.push_back(row.measurementVariance());
      out.rowPvar.push_back(row.parameterVariance());
      for(size_t j=0; j<KinKal::NParams(); ++j)
        out.rowDerivative.push_back(row.active() ? row.dRdP()[j] : std::numeric_limits<double>::quiet_NaN());
    }
    payload = std::move(out);
  }

  template<class KTRAJ> class HitOperandWriter {
    using Payload = KinKal::HitOperandTrace;
    TTree* tree_ = nullptr;
    int schema_ = 1, run_ = 0, subrun_ = 0, event_ = 0, iseed_ = 0;
    unsigned miter_ = 0, iter_ = 0;
    unsigned long long ordinal_ = 0, effect_ = 0;
    long long input_ = -1;
    bool active_ = false, inFit_ = false;
    double scale_ = std::numeric_limits<double>::quiet_NaN();
    Payload row_;
  public:
    void beginJob(art::TFileService& fs) {
      tree_ = fs.make<TTree>("hit_operands", "Value-owned straw-hit CA and residuals after updateState, before filtering");
      tree_->Branch("schema_version", &schema_);
      tree_->Branch("run", &run_); tree_->Branch("subrun", &subrun_);
      tree_->Branch("event", &event_); tree_->Branch("iseed", &iseed_);
      tree_->Branch("iteration_ordinal", &ordinal_); tree_->Branch("miter", &miter_); tree_->Branch("iter", &iter_);
      tree_->Branch("effect_index", &effect_); tree_->Branch("input_index", &input_);
      tree_->Branch("active", &active_); tree_->Branch("in_fit", &inFit_); tree_->Branch("variance_scale", &scale_);
      tree_->Branch("shindex", &row_.shindex); tree_->Branch("strawid", &row_.strawid);
      tree_->Branch("ca_status", &row_.caStatus); tree_->Branch("ca_usable", &row_.caUsable);
      tree_->Branch("precision", &row_.precision); tree_->Branch("hit_state", &row_.hitState);
      tree_->Branch("constrain_tot", &row_.constrainTOT); tree_->Branch("constrain_long", &row_.constrainLong);
      tree_->Branch("constrain_absdrift", &row_.constrainAbsDrift);
      tree_->Branch("particle_poca", &row_.particlePoca); tree_->Branch("sensor_poca", &row_.sensorPoca);
      tree_->Branch("particle_direction", &row_.particleDirection); tree_->Branch("sensor_direction", &row_.sensorDirection);
      tree_->Branch("doca", &row_.doca); tree_->Branch("doca_variance", &row_.docaVariance);
      tree_->Branch("toca_variance", &row_.tocaVariance); tree_->Branch("lsign", &row_.lsign);
      tree_->Branch("dDdP", &row_.dDdP); tree_->Branch("dTdP", &row_.dTdP);
      tree_->Branch("row_active", &row_.rowActive); tree_->Branch("row_value", &row_.rowValue);
      tree_->Branch("row_mvar", &row_.rowMvar); tree_->Branch("row_pvar", &row_.rowPvar);
      tree_->Branch("row_derivative", &row_.rowDerivative);
    }
    void write(int run, int subrun, int event, int seed, unsigned long long ordinal, size_t index,
               KinKal::IterationTrace<KTRAJ> const& trace, KinKal::EffectTrace<KTRAJ> const& effect) {
      if(!effect.hitOperands)return;
      if(!tree_ || effect.inputIndex == std::numeric_limits<size_t>::max())
        throw std::runtime_error("Invalid hit operand serialization identity");
      run_=run; subrun_=subrun; event_=event; iseed_=seed; ordinal_=ordinal; effect_=index;
      miter_=trace.status.miter_; iter_=trace.status.iter_; input_=effect.inputIndex;
      active_=effect.active; inFit_=effect.inFit; scale_=trace.varianceScale; row_=*effect.hitOperands;
      if(tree_->Fill() < 0)throw std::runtime_error("ROOT Fill failed: hit_operands");
    }
  };
}
#endif
