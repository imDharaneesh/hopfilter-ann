#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/common/types.hpp>
#include <hopfilter/graph/layer_assigner.hpp>

namespace hopfilter {
namespace {

constexpr std::size_t kMinNeighbors = 2;
constexpr int kDoubleMantissaBits = 53;
constexpr int kDiscardedBits = 64 - kDoubleMantissaBits;
constexpr double kMantissaScale =
    1.0 / static_cast<double>(std::uint64_t{1} << kDoubleMantissaBits);

double multiplierFor(std::size_t maxNeighbors) {
    if (maxNeighbors < kMinNeighbors) {
        throw InvalidArgumentError("maxNeighbors must be at least 2");
    }
    return 1.0 / std::log(static_cast<double>(maxNeighbors));
}

}

ExponentialLayerAssigner::ExponentialLayerAssigner(std::size_t maxNeighbors, std::uint64_t seed)
    : generator_(seed), levelMultiplier_(multiplierFor(maxNeighbors)) {}

LevelIndex ExponentialLayerAssigner::nextLevel() {
    const double level = std::floor(-std::log(nextOpenUnitInterval()) * levelMultiplier_);
    return static_cast<LevelIndex>(std::min(level, static_cast<double>(kMaxLevel)));
}

double ExponentialLayerAssigner::nextOpenUnitInterval() noexcept {
    return static_cast<double>((generator_() >> kDiscardedBits) + 1) * kMantissaScale;
}

}
