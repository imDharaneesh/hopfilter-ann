#include <algorithm>
#include <cstddef>
#include <span>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/common/types.hpp>
#include <hopfilter/search/search_context.hpp>
#include <hopfilter/search/search_layer.hpp>
#include <hopfilter/search/search_strategy.hpp>

namespace hopfilter {

SearchResults UnfilteredStrategy::search(const SearchContext& context,
                                         std::span<const float> query,
                                         std::size_t k,
                                         const SearchRequest& request) const {
    if (request.filter != nullptr) {
        throw InvalidArgumentError("the unfiltered strategy cannot apply a filter");
    }
    const auto start = descendToLevelZero(context, query);
    if (!start || k == 0) {
        return {};
    }
    auto results = searchLayer(context, query, {&*start, 1}, std::max(request.efSearch, k), 0);
    results.resize(std::min(results.size(), k));
    return results;
}

}
