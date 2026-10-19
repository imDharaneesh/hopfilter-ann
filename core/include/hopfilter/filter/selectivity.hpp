#pragma once

#include <cstddef>
#include <cstdint>

#include <hopfilter/filter/filter_predicate.hpp>

namespace hopfilter {

inline constexpr std::size_t kDefaultSelectivitySample = 1024;
inline constexpr std::uint64_t kDefaultSelectivitySeed = 0x5EEDF11E;

class SelectivityEstimator {
public:
    virtual ~SelectivityEstimator() = default;

    [[nodiscard]] virtual double estimate(const FilterPredicate& predicate,
                                          std::size_t universeSize) const = 0;
};

class ExactSelectivityEstimator final : public SelectivityEstimator {
public:
    [[nodiscard]] double estimate(const FilterPredicate& predicate,
                                  std::size_t universeSize) const override;
};

class SamplingSelectivityEstimator final : public SelectivityEstimator {
public:
    explicit SamplingSelectivityEstimator(std::size_t sampleSize = kDefaultSelectivitySample,
                                          std::uint64_t seed = kDefaultSelectivitySeed);

    [[nodiscard]] double estimate(const FilterPredicate& predicate,
                                  std::size_t universeSize) const override;

    [[nodiscard]] std::size_t sampleSize() const noexcept { return sampleSize_; }

private:
    std::size_t sampleSize_;
    std::uint64_t seed_;
};

}
