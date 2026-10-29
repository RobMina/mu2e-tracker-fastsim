#ifndef KinKal_PiecewiseCATrace_hh
#define KinKal_PiecewiseCATrace_hh

#include <limits>
#include <vector>

namespace KinKal {
  struct PiecewiseCATrace {
    bool failed = false;
    bool exhausted = false;
    unsigned long long initialIndex = 0;
    unsigned long long finalIndex = 0;
    unsigned long long solutionIndex = 0;
    unsigned long long maxMoves = 0;
    double particleHint = std::numeric_limits<double>::quiet_NaN();
    double sensorHint = std::numeric_limits<double>::quiet_NaN();
    double precision = std::numeric_limits<double>::quiet_NaN();
    std::vector<unsigned long long> solutionIndices;
    std::vector<double> inputParticleHints;
    std::vector<double> inputSensorHints;
    std::vector<double> particleTimes;
    std::vector<double> sensorTimes;
    std::vector<int> statuses;
    std::vector<bool> usable;

    void initialize(double hintparticle, double hintsensor, double prec,
        unsigned long long maxiter, unsigned long long startindex) noexcept {
      failed = false;
      exhausted = false;
      initialIndex = startindex;
      finalIndex = startindex;
      solutionIndex = startindex;
      maxMoves = maxiter;
      particleHint = hintparticle;
      sensorHint = hintsensor;
      precision = prec;
      solutionIndices.clear();
      inputParticleHints.clear();
      inputSensorHints.clear();
      particleTimes.clear();
      sensorTimes.clear();
      statuses.clear();
      usable.clear();
    }

    template<class CA> void solved(unsigned long long index, double hintparticle,
        double hintsensor, CA const& ca) noexcept {
      solutionIndex = index;
      if(failed) return;
      try {
        auto const status = static_cast<int>(ca.status());
        bool const isusable = ca.usable();
        double const pt = isusable ? ca.particleToca() : std::numeric_limits<double>::quiet_NaN();
        double const st = isusable ? ca.sensorToca() : std::numeric_limits<double>::quiet_NaN();
        auto const count = solutionIndices.size() + 1;
        // Reserve every column before appending so allocation failure leaves aligned rows.
        solutionIndices.reserve(count);
        inputParticleHints.reserve(count);
        inputSensorHints.reserve(count);
        particleTimes.reserve(count);
        sensorTimes.reserve(count);
        statuses.reserve(count);
        usable.reserve(count);
        solutionIndices.push_back(index);
        inputParticleHints.push_back(hintparticle);
        inputSensorHints.push_back(hintsensor);
        particleTimes.push_back(pt);
        sensorTimes.push_back(st);
        statuses.push_back(status);
        usable.push_back(isusable);
      } catch(...) {
        failed = true;
      }
    }
  };
}
#endif
