#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/common/types.hpp>
#include <hopfilter/filter/filter_predicate.hpp>
#include <hopfilter/search/filtered_search_config.hpp>
#include <hopfilter/search/filtered_strategies.hpp>
#include <hopfilter/search/search_context.hpp>
#include <hopfilter/search/search_layer.hpp>

namespace hopfilter {
namespace {

const FilterPredicate& requireFilter(const SearchRequest& request, const char* strategy) {
    if (request.filter == nullptr) {
        throw InvalidArgumentError(std::string{strategy} + " search needs a filter");
    }
    return *request.filter;
}

void keepClosest(SearchResults& heap, const Neighbor& candidate, std::size_t k) {
    if (heap.size() < k) {
        heap.push_back(candidate);
        std::ranges::push_heap(heap, CloserFirst{});
        return;
    }
    if (CloserFirst{}(candidate, heap.front())) {
        std::ranges::pop_heap(heap, CloserFirst{});
        heap.back() = candidate;
        std::ranges::push_heap(heap, CloserFirst{});
    }
}

}

PostFilterStrategy::PostFilterStrategy(double overFetchFactor) : overFetchFactor_(overFetchFactor) {
    if (!(overFetchFactor >= 1.0)) {
        throw InvalidArgumentError("post-filter over-fetch factor must be at least 1");
    }
}

SearchResults PostFilterStrategy::search(const SearchContext& context,
                                         std::span<const float> query,
                                         std::size_t k,
                                         const SearchRequest& request) const {
    const FilterPredicate& filter = requireFilter(request, "post-filter");
    const auto start = descendToLevelZero(context, query);
    if (!start || k == 0) {
        return {};
    }
    const auto overFetched =
        static_cast<std::size_t>(std::ceil(static_cast<double>(k) * overFetchFactor_));
    auto results =
        searchLayer(context, query, {&*start, 1}, std::max(request.efSearch, overFetched), 0);
    std::erase_if(results,
                  [&filter](const Neighbor& neighbor) { return !filter.matches(neighbor.id); });
    results.resize(std::min(results.size(), k));
    return results;
}

SearchResults PreFilterStrategy::search(const SearchContext& context,
                                        std::span<const float> query,
                                        std::size_t k,
                                        const SearchRequest& request) const {
    const FilterPredicate& filter = requireFilter(request, "pre-filter");
    if (k == 0) {
        return {};
    }
    std::vector<NodeId> matches;
    filter.collectMatches(context.graph().size(), matches);
    SearchResults closest;
    closest.reserve(std::min(k, matches.size()) + 1);
    for (std::size_t position = 0; position < matches.size(); ++position) {
        if (position + 1 < matches.size()) {
            context.distances().prefetch(matches[position + 1]);
        }
        const NodeId id = matches[position];
        keepClosest(closest, {id, context.distances().toQuery(query, id)}, k);
    }
    std::ranges::sort(closest, CloserFirst{});
    return closest;
}

InGraphStrategy::InGraphStrategy(const FilteredSearchConfig& config)
    : rejectionWindow_(config.rejectionWindow),
      rejectionThreshold_(config.rejectionThreshold),
      efGrowthFactor_(config.efGrowthFactor),
      maxEf_(config.maxEf) {
    validate(config);
}

SearchResults InGraphStrategy::search(const SearchContext& context,
                                      std::span<const float> query,
                                      std::size_t k,
                                      const SearchRequest& request) const {
    const FilterPredicate& filter = requireFilter(request, "in-graph");
    const auto start = descendToLevelZero(context, query);
    if (!start || k == 0) {
        return {};
    }
    auto results = filteredSearchLayer(
        context, query, {&*start, 1}, filter, scheduleFor(std::max(request.efSearch, k)));
    results.resize(std::min(results.size(), k));
    return results;
}

EfSchedule InGraphStrategy::scheduleFor(std::size_t ef) const noexcept {
    return {ef, rejectionWindow_, rejectionThreshold_, efGrowthFactor_, std::max(maxEf_, ef)};
}

}
