#ifndef TrkFastSim_CrossingGeometryWriter_hh
#define TrkFastSim_CrossingGeometryWriter_hh

#include "Offline/TrackerGeom/inc/Tracker.hh"
#include "art_root_io/TFileService.h"
#include "TTree.h"
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace mu2e {
// Standalone diagnostic trees; no changes to any pre-existing schema. Call only
// for selected targets, BEFORE build/fit. Every call copies current conditions:
// no pointer cache, address identity, assumed IOV immutability or lossy hash.
// This deliberately trades output size (~3M straw rows/147 targets) for bounded
// memory and unambiguous conditions binding. Check output quota before running.
// A snapshot is usable ONLY with status=0 followed by status=1 in bindings,
// matching counts and contiguous ordinals. Missing/failed terminal rows invalidate
// the target. Partial ROOT writes cannot be rolled back. All Fill failures throw.
// Coordinates are native DS mm; doubles are copied without normalization. Native
// float halfLength is stored as float. Rotation layout is row-major; p'=R*p+t.
// No physics, trajectory, PCA, RNG or fit-state accessor is called here.
class CrossingGeometryWriter {
  bool ready_ = false;
  TTree *bindings_ = nullptr, *tracker_ = nullptr, *planes_ = nullptr;
  TTree *panels_ = nullptr, *straws_ = nullptr;
  int schema_ = 1, run_ = 0, subrun_ = 0, event_ = 0, iseed_ = 0;
  unsigned long long next_ = 0, geometry_ = 0;
  int status_ = 0; // 0 attempted, 1 complete, -1 failed; not a physics status
  std::string error_;
  unsigned long long nplanes_ = 0, npanels_ = 0, nstraws_ = 0;
  unsigned long long planeOrdinal_ = 0, panelOrdinal_ = 0, strawOrdinal_ = 0;
  unsigned long long trackerPanelOrdinal_ = 0, trackerStrawOrdinal_ = 0;
  unsigned fullsid_ = 0, planeId_ = 0, panelId_ = 0;
  bool exists_ = false;
  float halfLength_ = 0;
  double innerRadius_ = 0, outerRadius_ = 0, wireRadius_ = 0;
  std::vector<unsigned> children_;
  std::vector<double> origin_, u_, v_, w_, rotation_, translation_;
  std::vector<double> tubeCal_, tubeHV_, tubeMid_, tubeDirection_;
  std::vector<double> wireCal_, wireHV_, wireMid_, wireDirection_;

  template<class V> static void xyz(std::vector<double>& out, V const& p) {
    out = {p.x(), p.y(), p.z()};
  }
  static void fill(TTree* tree) {
    if (!tree || tree->Fill() < 0)
      throw std::runtime_error("Crossing geometry ROOT Fill failed");
  }
  template<class T> static void branch(TTree* tree, char const* name, T* value) {
    if (!tree->Branch(name, value))
      throw std::runtime_error("Crossing geometry ROOT Branch failed");
  }
  void common(TTree* tree) {
    branch(tree, "schema_version", &schema_);
    branch(tree, "geometry_index", &geometry_);
  }
  void identity(TTree* tree) {
    branch(tree, "fullsid", &fullsid_);
    branch(tree, "plane_id", &planeId_);
    branch(tree, "plane_ordinal", &planeOrdinal_);
  }
  void frame(TTree* tree) {
    branch(tree, "origin", &origin_);
    branch(tree, "u_direction", &u_);
    branch(tree, "v_direction", &v_);
    branch(tree, "w_direction", &w_);
  }
  template<class G> void copyFrame(G const& object) {
    xyz(origin_, object.origin()); xyz(u_, object.uDirection());
    xyz(v_, object.vDirection()); xyz(w_, object.wDirection());
  }
  void transform(HepTransform const& transform) {
    // HepTransform is mu2e's wrapper, NOT CLHEP::HepGeom::Transform3D.
    // HepTransform.cc applies displacement()+rotation()*p. Copy native inverse
    // coefficients, never reconstruct by transformed basis points/subtraction.
    auto const& r = transform.rotation();
    rotation_ = {r.xx(), r.xy(), r.xz(), r.yx(), r.yy(), r.yz(),
                 r.zx(), r.zy(), r.zz()};
    xyz(translation_, transform.displacement());
  }
  static void require(bool condition, char const* message) {
    if (!condition) throw std::runtime_error(message);
  }

 public:
  CrossingGeometryWriter() = default;
  CrossingGeometryWriter(CrossingGeometryWriter const&) = delete;
  CrossingGeometryWriter& operator=(CrossingGeometryWriter const&) = delete;

  void beginJob(art::TFileService& fs) {
    require(!bindings_, "Crossing geometry beginJob called twice");
    bindings_ = fs.make<TTree>("crossing_geometry_bindings", "Selected target geometry attempt and terminal status");
    tracker_ = fs.make<TTree>("crossing_geometry_tracker", "Native tracker snapshot metadata");
    planes_ = fs.make<TTree>("crossing_geometry_planes", "Native ordered planes including absent planes");
    panels_ = fs.make<TTree>("crossing_geometry_panels", "Native ordered panels and exact dsToPanel coefficients");
    straws_ = fs.make<TTree>("crossing_geometry_straws", "Native ordered tube and wire geometry");
    for (auto tree : {bindings_, tracker_, planes_, panels_, straws_}) common(tree);
    branch(bindings_, "run", &run_); branch(bindings_, "subrun", &subrun_);
    branch(bindings_, "event", &event_); branch(bindings_, "iseed", &iseed_);
    branch(bindings_, "source_status", &status_); branch(bindings_, "error", &error_);
    branch(bindings_, "nplanes", &nplanes_); branch(bindings_, "npanels", &npanels_);
    branch(bindings_, "nstraws", &nstraws_);
    branch(tracker_, "origin", &origin_);
    branch(tracker_, "straw_inner_radius", &innerRadius_);
    branch(tracker_, "straw_outer_radius", &outerRadius_);
    branch(tracker_, "wire_radius", &wireRadius_);
    identity(planes_); frame(planes_);
    branch(planes_, "plane_exists", &exists_);
    branch(planes_, "ordered_panel_fullsid", &children_);
    identity(panels_); frame(panels_);
    branch(panels_, "panel_id", &panelId_);
    branch(panels_, "panel_ordinal", &panelOrdinal_);
    branch(panels_, "tracker_panel_ordinal", &trackerPanelOrdinal_);
    branch(panels_, "ordered_straw_fullsid", &children_);
    branch(panels_, "ds_to_panel_rotation", &rotation_);
    branch(panels_, "ds_to_panel_translation", &translation_);
    identity(straws_);
    branch(straws_, "panel_id", &panelId_);
    branch(straws_, "panel_ordinal", &panelOrdinal_);
    branch(straws_, "straw_ordinal", &strawOrdinal_);
    branch(straws_, "tracker_straw_ordinal", &trackerStrawOrdinal_);
    branch(straws_, "half_length", &halfLength_);
    branch(straws_, "tube_cal", &tubeCal_); branch(straws_, "tube_hv", &tubeHV_);
    branch(straws_, "tube_midpoint", &tubeMid_); branch(straws_, "tube_direction", &tubeDirection_);
    branch(straws_, "wire_cal", &wireCal_); branch(straws_, "wire_hv", &wireHV_);
    branch(straws_, "wire_midpoint", &wireMid_); branch(straws_, "wire_direction", &wireDirection_);
    ready_ = true;
  }

  void write(Tracker const& tracker, int run, int subrun, int event, int iseed) {
    require(ready_ && bindings_ && tracker_ && planes_ && panels_ && straws_, "Crossing geometry not initialized");
    geometry_ = next_++; run_ = run; subrun_ = subrun; event_ = event; iseed_ = iseed;
    nplanes_ = tracker.nPlanes(); npanels_ = tracker.nPanels(); nstraws_ = tracker.nStraws();
    status_ = 0; error_.clear();
    try {
      fill(bindings_);
      xyz(origin_, tracker.origin());
      innerRadius_ = tracker.strawInnerRadius(); outerRadius_ = tracker.strawOuterRadius();
      wireRadius_ = tracker.wireRadius(); fill(tracker_);
      unsigned long long panelCount = 0, strawCount = 0;
      // Tiny hierarchy membership buffers (216 panels/20736 straws), not geometry
      // caches. Reject aliasing, omissions and inconsistent flat/hierarchy storage.
      std::vector<bool> seenPanels(npanels_, false), seenStraws(nstraws_, false);
      planeOrdinal_ = 0;
      for (auto const& plane : tracker.planes()) {
        planeId_ = plane.id().plane(); fullsid_ = plane.id().asUint16();
        require(planeId_ == planeOrdinal_, "Crossing geometry plane ordering differs");
        exists_ = tracker.planeExists(plane.id()); copyFrame(plane);
        children_.clear();
        for (auto const* panel : plane.panels()) {
          require(panel, "Null geometry panel"); children_.push_back(panel->id().asUint16());
        }
        fill(planes_); panelOrdinal_ = 0;
        for (auto const* panel : plane.panels()) {
          auto const sid = panel->id(); panelId_ = sid.panel(); fullsid_ = sid.asUint16();
          trackerPanelOrdinal_ = sid.uniquePanel();
          require(sid.plane() == planeId_ && trackerPanelOrdinal_ < npanels_, "Geometry panel identity differs");
          require(!seenPanels[trackerPanelOrdinal_], "Duplicate geometry panel");
          seenPanels[trackerPanelOrdinal_] = true;
          require(panel == &tracker.panels().at(trackerPanelOrdinal_), "Unsupported non-owning tracker panel hierarchy");
          copyFrame(*panel); transform(panel->dsToPanel()); children_.clear();
          for (auto const* straw : panel->straws()) {
            require(straw, "Null geometry straw"); children_.push_back(straw->id().asUint16());
          }
          fill(panels_); ++panelCount; strawOrdinal_ = 0;
          for (auto const* straw : panel->straws()) {
            auto const id = straw->id(); fullsid_ = id.asUint16(); trackerStrawOrdinal_ = id.uniqueStraw();
            require(id.plane() == planeId_ && id.panel() == panelId_ && trackerStrawOrdinal_ < nstraws_, "Geometry straw identity differs");
            require(!seenStraws[trackerStrawOrdinal_], "Duplicate geometry straw");
            seenStraws[trackerStrawOrdinal_] = true;
            require(straw == &tracker.straws().at(trackerStrawOrdinal_), "Unsupported non-owning tracker straw hierarchy");
            halfLength_ = straw->halfLength();
            xyz(tubeCal_, straw->strawEnd(StrawEnd::cal)); xyz(tubeHV_, straw->strawEnd(StrawEnd::hv));
            xyz(tubeMid_, straw->strawPosition()); xyz(tubeDirection_, straw->strawDirection());
            xyz(wireCal_, straw->wireEnd(StrawEnd::cal)); xyz(wireHV_, straw->wireEnd(StrawEnd::hv));
            xyz(wireMid_, straw->getMidPoint()); xyz(wireDirection_, straw->wireDirection());
            fill(straws_); ++strawCount; ++strawOrdinal_;
          }
          ++panelOrdinal_;
        }
        ++planeOrdinal_;
      }
      require(planeOrdinal_ == nplanes_ && panelCount == npanels_ && strawCount == nstraws_, "Incomplete crossing geometry hierarchy");
      status_ = 1; fill(bindings_);
    } catch (...) {
      // Best-effort independent failure evidence. Even if this Fill fails, a
      // missing complete binding makes the external capture gate fail closed.
      auto failure = std::current_exception(); status_ = -1;
      try {
        try { std::rethrow_exception(failure); }
        catch (std::exception const& e) { error_ = e.what(); }
        catch (...) { error_ = "Unknown crossing geometry failure"; }
        fill(bindings_);
      } catch (...) {}
      std::rethrow_exception(failure);
    }
  }
};
} // namespace mu2e
#endif
