#pragma once

#include <cstddef>

#include <hopfilter/common/types.hpp>
#include <hopfilter/filter/filter_predicate.hpp>
#include <hopfilter/search/search_context.hpp>

namespace hopfilter {

[[nodiscard]] double neighbourhoodShare(const SearchContext& context,
                                        NodeId start,
                                        const FilterPredicate& filter,
                                        std::size_t probeSize);

}
