#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include <hopfilter/common/types.hpp>
#include <hopfilter/filter/attribute_store.hpp>
#include <hopfilter/filter/filter_predicate.hpp>
#include <hopfilter/index/hnsw_index.hpp>
#include <hopfilter/index/index_config.hpp>

#include "support/brute_force.hpp"
#include "support/random_vectors.hpp"

namespace hopfilter::testing {

inline constexpr std::size_t kFilteredPoints = 3000;
inline constexpr std::size_t kFilteredDimension = 16;
inline constexpr CategoryCode kBuckets = 100;

class FilteredIndex {
public:
    explicit FilteredIndex(std::uint64_t seed) {
        RandomVectors random{seed};
        data_ = random.vectors(kFilteredPoints, kFilteredDimension);
        queries_ = random.vectors(50, kFilteredDimension);
        IndexConfig config;
        config.maxNeighbors = 12;
        config.efConstruction = 100;
        index_ = buildIndex(data_, config);
        CategoricalColumn buckets(kFilteredPoints);
        for (std::size_t id = 0; id < kFilteredPoints; ++id) {
            buckets[id] = static_cast<CategoryCode>(id % kBuckets);
        }
        index_->setCategoricalAttribute("bucket", std::move(buckets));
    }

    [[nodiscard]] const HnswIndex& index() const { return *index_; }
    [[nodiscard]] HnswIndex& index() { return *index_; }
    [[nodiscard]] const std::vector<std::vector<float>>& data() const { return data_; }
    [[nodiscard]] const std::vector<std::vector<float>>& queries() const { return queries_; }

    [[nodiscard]] std::vector<NodeId> truth(std::span<const float> query,
                                            std::size_t k,
                                            const FilterPredicate& filter) const {
        std::vector<std::vector<float>> matching;
        std::vector<NodeId> ids;
        for (std::size_t id = 0; id < data_.size(); ++id) {
            if (filter.matches(static_cast<NodeId>(id))) {
                matching.push_back(data_[id]);
                ids.push_back(static_cast<NodeId>(id));
            }
        }
        if (matching.empty()) {
            return {};
        }
        auto local = bruteForceTopK(matching, query, k, index_->metric());
        std::ranges::transform(
            local, local.begin(), [&ids](NodeId position) { return ids[position]; });
        return local;
    }

private:
    std::vector<std::vector<float>> data_;
    std::vector<std::vector<float>> queries_;
    std::unique_ptr<HnswIndex> index_;
};

inline bool isWellFormed(const SearchResults& results, const FilterPredicate& filter) {
    std::unordered_set<NodeId> seen;
    for (std::size_t position = 0; position < results.size(); ++position) {
        if (!filter.matches(results[position].id) || !seen.insert(results[position].id).second) {
            return false;
        }
        if (position > 0 && CloserFirst{}(results[position], results[position - 1])) {
            return false;
        }
    }
    return true;
}

}
