#pragma once

#include <cstddef>
#include <cstdint>
#include <random>

#include <hopfilter/common/types.hpp>

namespace hopfilter {

class LayerAssigner {
public:
    virtual ~LayerAssigner() = default;

    [[nodiscard]] virtual LevelIndex nextLevel() = 0;
};

class ExponentialLayerAssigner final : public LayerAssigner {
public:
    static constexpr LevelIndex kMaxLevel = 32;

    ExponentialLayerAssigner(std::size_t maxNeighbors, std::uint64_t seed);

    [[nodiscard]] LevelIndex nextLevel() override;

    [[nodiscard]] double levelMultiplier() const noexcept { return levelMultiplier_; }

private:
    [[nodiscard]] double nextOpenUnitInterval() noexcept;

    std::mt19937_64 generator_;
    double levelMultiplier_;
};

}
