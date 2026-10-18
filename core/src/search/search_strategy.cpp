#include <algorithm>
#include <cstddef>
#include <span>

#include <hopfilter/common/types.hpp>
#include <hopfilter/search/search_context.hpp>
#include <hopfilter/search/search_layer.hpp>
#include <hopfilter/search/search_strategy.hpp>

namespace hopfilter {

SearchResults UnfilteredStrategy::search(const SearchContext& context,
                                         std::span<const float> query,
                                         std::size_t k,
                                         const SearchRequest& request) const {
    const auto entryPoint = context.graph().entryPoint();
    if (!entryPoint || k == 0) {
        return {};
    }
    const Neighbor entry{*entryPoint, context.distances().toQuery(query, *entryPoint)};
    const Neighbor start = greedyDescend(context, query, entry, context.graph().maxLevel(), 0);
    auto results = searchLayer(context, query, {&start, 1}, std::max(request.efSearch, k), 0);
    results.resize(std::min(results.size(), k));
    return results;
}

}
