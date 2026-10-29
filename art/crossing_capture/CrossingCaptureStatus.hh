#ifndef TrkFastSim_CrossingCaptureStatus_hh
#define TrkFastSim_CrossingCaptureStatus_hh
#include "KinKal/Fit/CrossingTrace.hh"
#include "art_root_io/TFileService.h"
#include "TTree.h"
#include <stdexcept>
#include <string>

namespace mu2e {
  class CrossingCaptureStatus {
    TTree* tree_=nullptr;
    bool ready_=false;
    template<class T> void branch(char const* name,T* value) {
      if(!tree_ || !tree_->Branch(name,value))throw std::runtime_error("Crossing diagnostic status Branch failed");
    }
    int schema_=1, run_=0, subrun_=0, event_=0, iseed_=0, status_=0;
    unsigned long long records_=0, objects_=0, trajectories_=0, discovery_=0;
    std::string error_;
  public:
    void beginJob(art::TFileService& fs) {
      tree_=fs.make<TTree>("crossing_capture_status","Crossing diagnostic attempt and serialization completion");
      branch("schema_version",&schema_); branch("run",&run_); branch("subrun",&subrun_);
      branch("event",&event_); branch("iseed",&iseed_); branch("status",&status_);
      branch("n_records",&records_); branch("n_objects",&objects_);
      branch("n_trajectories",&trajectories_); branch("n_discovery",&discovery_); branch("error",&error_);
      ready_=true;
    }
    void write(int run,int subrun,int event,int iseed,int status,KinKal::CrossingTraceSink const& sink,std::string const& error) {
      run_=run; subrun_=subrun; event_=event; iseed_=iseed; status_=status; error_=error;
      records_=sink.records.size(); objects_=sink.constructors.size(); trajectories_=sink.trajectories.size(); discovery_=sink.discovery.size();
      if(!ready_ || !tree_ || tree_->Fill()<0)throw std::runtime_error("Crossing diagnostic status Fill failed");
    }
  };
}
#endif
