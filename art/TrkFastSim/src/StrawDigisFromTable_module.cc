// Replay detector digis and their ordered ADC waveforms without synthesising MC truth.
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "TBranch.h"
#include "TFile.h"
#include "TLeaf.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"
#include "TTreeReaderValue.h"
#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Principal/Event.h"
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"
#include "Offline/RecoDataProducts/inc/StrawDigi.hh"

namespace mu2e {
namespace {
using Key = std::tuple<unsigned, unsigned, unsigned>;

class IntegerArray {
 public:
  virtual ~IntegerArray() = default;
  virtual std::vector<std::uint64_t> values() = 0;
};

template <class T> class TypedArray : public IntegerArray {
 public:
  TypedArray(TTreeReader& reader, const char* name) : array_(reader, name) {}
  std::vector<std::uint64_t> values() override {
    if (array_.GetSetupStatus() < 0) throw cet::exception("TRKDIGITABLE") << "unreadable integer array";
    std::vector<std::uint64_t> out;
    out.reserve(array_.GetSize());
    for (auto value : array_) {
      if (value < 0) throw cet::exception("TRKDIGITABLE") << "negative digi value";
      out.push_back(static_cast<std::uint64_t>(value));
    }
    return out;
  }
 private:
  TTreeReaderArray<T> array_;
};

std::unique_ptr<IntegerArray> integerArray(TTree* tree, TTreeReader& reader, const char* name) {
  auto* branch = tree->GetBranch(name);
  if (!branch) throw cet::exception("TRKDIGITABLE") << "missing branch " << name;
  std::string type = branch->GetClassName();
  auto* leaf = branch->GetLeaf(name);
  if (type.empty() && leaf && leaf->GetLeafCount()) type = leaf->GetTypeName();
  if (type == "vector<int>" || type == "std::vector<int>" || type == "Int_t")
    return std::make_unique<TypedArray<Int_t>>(reader, name);
  if (type == "vector<unsigned int>" || type == "std::vector<unsigned int>" || type == "UInt_t")
    return std::make_unique<TypedArray<UInt_t>>(reader, name);
  throw cet::exception("TRKDIGITABLE") << "unsupported array type " << type << " for " << name;
}

void bound(const std::vector<std::uint64_t>& values, std::uint64_t max, const char* name) {
  for (auto value : values)
    if (value > max) throw cet::exception("TRKDIGITABLE") << name << " out of range: " << value;
}
}  // namespace

class StrawDigisFromTable : public art::EDProducer {
 public:
  struct Config {
    fhicl::Atom<std::string> tableFile{fhicl::Name("tableFile"), fhicl::Comment("lossless digi table")};
    fhicl::Atom<std::string> directory{fhicl::Name("directory"), fhicl::Comment("event tree directory"), "trkdigi"};
  };
  using Parameters = art::EDProducer::Table<Config>;
  explicit StrawDigisFromTable(Parameters const& config);
  void produce(art::Event& event) override;
 private:
  std::unique_ptr<TFile> file_;
  std::unique_ptr<TTreeReader> reader_;
  std::map<Key, Long64_t> rows_;
  std::map<std::string, std::unique_ptr<IntegerArray>> columns_;
};

StrawDigisFromTable::StrawDigisFromTable(Parameters const& config) : art::EDProducer(config) {
  produces<StrawDigiCollection>();
  produces<StrawDigiADCWaveformCollection>();
  file_.reset(TFile::Open(config().tableFile().c_str(), "READ"));
  if (!file_ || file_->IsZombie()) throw cet::exception("TRKDIGITABLE") << "cannot open " << config().tableFile();
  auto* tree = file_->Get<TTree>((config().directory() + "/events").c_str());
  if (!tree) throw cet::exception("TRKDIGITABLE") << "missing events TTree";
  for (auto name : {"run", "subrun", "event"}) {
    auto* leaf = tree->GetLeaf(name);
    if (!leaf || std::string(leaf->GetTypeName()) != "UInt_t" || leaf->GetLeafCount() || leaf->GetLenStatic() != 1)
      throw cet::exception("TRKDIGITABLE") << "event key must be a scalar uint32: " << name;
  }
  {
    TTreeReader index(tree);
    TTreeReaderValue<UInt_t> run(index, "run"), subrun(index, "subrun"), event(index, "event");
    while (index.Next()) {
      if (run.GetSetupStatus() < 0 || subrun.GetSetupStatus() < 0 || event.GetSetupStatus() < 0)
        throw cet::exception("TRKDIGITABLE") << "event keys must be uint32";
      if (!rows_.emplace(Key(*run, *subrun, *event), index.GetCurrentEntry()).second)
        throw cet::exception("TRKDIGITABLE") << "duplicate event identity in table";
    }
    if (index.GetEntryStatus() != TTreeReader::kEntryBeyondEnd)
      throw cet::exception("TRKDIGITABLE") << "cannot index event identities";
  }
  reader_ = std::make_unique<TTreeReader>(tree);
  for (auto name : {"d_sid", "d_tdc0", "d_tdc1", "d_tot0", "d_tot1", "d_pmp", "d_flags", "d_nadc", "d_adc"})
    columns_.emplace(name, integerArray(tree, *reader_, name));
}

void StrawDigisFromTable::produce(art::Event& event) {
  auto row = rows_.find(Key(event.run(), event.subRun(), event.event()));
  if (row == rows_.end()) throw cet::exception("TRKDIGITABLE") << "event absent from table: " << event.id();
  if (reader_->SetEntry(row->second) != TTreeReader::kEntryValid)
    throw cet::exception("TRKDIGITABLE") << "unreadable event " << event.id();
  std::map<std::string, std::vector<std::uint64_t>> data;
  for (auto& column : columns_) data.emplace(column.first, column.second->values());
  auto const n = data.at("d_sid").size();
  for (auto const& column : data)
    if (column.first != "d_adc" && column.second.size() != n)
      throw cet::exception("TRKDIGITABLE") << "inconsistent digi count for " << column.first << " in " << event.id();
  for (auto name : {"d_tdc0", "d_tdc1"}) bound(data.at(name), std::numeric_limits<TrkTypes::TDCValue>::max(), name);
  for (auto name : {"d_sid", "d_tot0", "d_tot1", "d_pmp", "d_adc"}) bound(data.at(name), 65535, name);
  bound(data.at("d_flags"), 255, "d_flags");
  std::size_t samples = 0;
  for (auto count : data.at("d_nadc")) {
    if (count > data.at("d_adc").size() - samples)
      throw cet::exception("TRKDIGITABLE") << "ADC count overflow in " << event.id();
    samples += count;
  }
  if (samples != data.at("d_adc").size()) throw cet::exception("TRKDIGITABLE") << "extra ADC samples in " << event.id();
  auto digis = std::make_unique<StrawDigiCollection>();
  auto waveforms = std::make_unique<StrawDigiADCWaveformCollection>();
  digis->reserve(n);
  waveforms->reserve(n);
  std::size_t offset = 0;
  for (std::size_t i = 0; i < n; ++i) {
    StrawId sid(static_cast<std::uint16_t>(data.at("d_sid")[i]));
    if (!sid.valid()) throw cet::exception("TRKDIGITABLE") << "invalid straw ID in " << event.id();
    TrkTypes::TDCValues tdc{{static_cast<TrkTypes::TDCValue>(data.at("d_tdc0")[i]), static_cast<TrkTypes::TDCValue>(data.at("d_tdc1")[i])}};
    TrkTypes::TOTValues tot{{static_cast<TrkTypes::TOTValue>(data.at("d_tot0")[i]), static_cast<TrkTypes::TOTValue>(data.at("d_tot1")[i])}};
    digis->emplace_back(sid, tdc, tot, static_cast<TrkTypes::ADCValue>(data.at("d_pmp")[i]));
    for (unsigned bit = 0; bit < 8; ++bit)
      if (data.at("d_flags")[i] & (1u << bit)) digis->back().digiFlag().merge(static_cast<StrawDigiFlag::bit_type>(bit));
    TrkTypes::ADCWaveform adc;
    auto count = data.at("d_nadc")[i];
    adc.reserve(count);
    for (std::size_t j = 0; j < count; ++j) adc.push_back(static_cast<TrkTypes::ADCValue>(data.at("d_adc")[offset++]));
    waveforms->emplace_back(adc);
  }
  event.put(std::move(digis));
  event.put(std::move(waveforms));
}
}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::StrawDigisFromTable)
