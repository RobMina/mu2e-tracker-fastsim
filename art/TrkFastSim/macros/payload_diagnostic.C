// Decode saved seed payloads through compiled dictionaries without normalizing persistent fields.
#include "TBufferFile.h"
#include "TClass.h"
#include "TFile.h"
#include "TInterpreter.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TSystem.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TVirtualCollectionProxy.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

int payload_offset(TClass* cls, const char* name) {
  auto* info = cls->GetStreamerInfo();
  auto* member = info ? dynamic_cast<TStreamerElement*>(info->GetElements()->FindObject(name)) : nullptr;
  if (!member) throw std::runtime_error(std::string("missing member ") + name);
  return member->GetOffset();
}

void payload_diagnostic(const char* reference, const char* candidate) {
  gInterpreter->SetClassAutoLoading(false);
  gSystem->Load("libmu2e_DataProducts_dict");
  gSystem->Load("libmu2e_RecoDataProducts_dict");
  auto* cls = TClass::GetClass("std::vector<mu2e::CosmicTrackSeed>");
  auto* seed = TClass::GetClass("mu2e::CosmicTrackSeed");
  auto* track = TClass::GetClass("mu2e::CosmicTrack");
  auto* params = TClass::GetClass("mu2e::TrackParams");
  if (!cls || !seed || !track || !params) throw std::runtime_error("missing seed dictionaries");
  auto* pa = cls->GetCollectionProxy();
  auto* pb = pa->Generate();
  TFile left(reference), right(candidate);
  if (left.IsZombie() || right.IsZombie()) throw std::runtime_error("missing payload input");
  TTreeReader a("payload/events", &left), b("payload/events", &right);
  TTreeReaderValue<unsigned> run(a, "run"), subrun(a, "subrun"), event(a, "event");
  TTreeReaderValue<std::vector<unsigned char>> aa(a, "line_seeds"), bb(b, "line_seeds");
  while (a.Next()) {
    if (!b.Next()) throw std::runtime_error("different event counts");
    if (*aa == *bb) continue;
    TBufferFile ba(TBuffer::kRead, aa->size(), aa->data(), false), bc(TBuffer::kRead, bb->size(), bb->data(), false);
    ba.SetParent(&left); bc.SetParent(&right);
    void* sa = ba.ReadObjectAny(cls);
    void* sb = bc.ReadObjectAny(cls);
    if (!sa || !sb) throw std::runtime_error("seed deserialization failure");
    {
      TVirtualCollectionProxy::TPushPop ga(pa, sa), gb(pb, sb);
      if (pa->Size() != pb->Size()) throw std::runtime_error("different seed counts");
      for (unsigned i = 0; i < pa->Size(); ++i) {
        const char* x = static_cast<const char*>(pa->At(i)) + payload_offset(seed, "_track");
        const char* y = static_cast<const char*>(pb->At(i)) + payload_offset(seed, "_track");
        std::cout << "event=" << *run << ':' << *subrun << ':' << *event << " seed=" << i << '\n';
        auto compare = [](const char* group, const char* name, const char* l, const char* r, int offset) {
          double u, v;
          std::memcpy(&u, l + offset, sizeof(u)); std::memcpy(&v, r + offset, sizeof(v));
          if (std::memcmp(&u, &v, sizeof(u)) != 0)
            std::cout << "different " << group << '.' << name << " left=" << std::hexfloat << u << " right=" << v << std::defaultfloat << '\n';
        };
        for (const char* group : {"InitParams", "FitParams", "MinuitParams"}) {
          int offset = payload_offset(track, group);
          for (const char* name : {"A0", "A1", "B0", "B1", "T0", "deltaA0", "deltaA1", "deltaB0", "deltaB1", "deltaT0"})
            compare(group, name, x + offset, y + offset, payload_offset(params, name));
        }
        for (const char* name : {"d0_", "z0_", "t0_", "cost_", "phi0_", "mom_"})
          compare("track", name, x, y, payload_offset(track, name));
        int u, v, offset = payload_offset(track, "niters");
        std::memcpy(&u, x + offset, sizeof(u)); std::memcpy(&v, y + offset, sizeof(v));
        if (u != v) std::cout << "different track.niters left=" << u << " right=" << v << '\n';
      }
    }
    cls->Destructor(sa); cls->Destructor(sb);
  }
  if (b.Next()) throw std::runtime_error("different event counts");
  delete pb;
}
