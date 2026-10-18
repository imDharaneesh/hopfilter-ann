#pragma once

#include <cstddef>
#include <cstdint>
#include <random>

#include <hopfilter/common/types.hpp>

namespace hopfilter {

/// Chooses the top layer of each newly inserted node. Not thread-safe; the index
/// serializes inserts.
class LayerAssigner {
public:
    virtual ~LayerAssigner() = default;

    [[nodiscard]] virtual LevelIndex nextLevel() = 0;
};

/// The HNSW paper's level distribution: ``floor(-ln(U) * mL)`` with ``mL = 1 / ln(M)``,
/// so a node reaches level ``l`` with probability ``M^-l``.
///
/// Uniform draws are built from raw generator bits rather than a standard library
/// distribution, so the same seed yields the same graph on every platform.
class ExponentialLayerAssigner final : public LayerAssigner {
public:
    static constexpr LevelIndex kMaxLevel = 32;

    /// Throws InvalidArgumentError if ``maxNeighbors`` is below 2.
    ExponentialLayerAssigner(std::size_t maxNeighbors, std::uint64_t seed);

    [[nodiscard]] LevelIndex nextLevel() override;

    [[nodiscard]] double levelMultiplier() const noexcept { return levelMultiplier_; }

private:
    [[nodiscard]] double nextOpenUnitInterval() noexcept;

    std::mt19937_64 generator_;
    double levelMultiplier_;
};

}
