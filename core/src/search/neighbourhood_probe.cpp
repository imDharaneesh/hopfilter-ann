#include <cstddef>
#include <vector>

#include <hopfilter/common/types.hpp>
#include <hopfilter/filter/filter_predicate.hpp>
#include <hopfilter/search/neighbourhood_probe.hpp>
#include <hopfilter/search/search_context.hpp>

namespace hopfilter {

double neighbourhoodShare(const SearchContext& context,
                          NodeId start,
                          const FilterPredicate& filter,
                          std::size_t probeSize) {
    VisitedSet& visited = context.visited();
    visited.clear();
    static_cast<void>(visited.markVisited(start));
    std::vector<NodeId> frontier{start};
    std::size_t probed = 0;
    std::size_t matching = 0;
    for (std::size_t next = 0; next < frontier.size() && probed < probeSize; ++next) {
        const NodeId node = frontier[next];
        ++probed;
        matching += filter.matches(node) ? std::size_t{1} : std::size_t{0};
        for (const NodeId neighbour : context.graph().neighbors(node, 0)) {
            if (frontier.size() < probeSize && visited.markVisited(neighbour)) {
                frontier.push_back(neighbour);
            }
        }
    }
    return probed == 0 ? 0.0 : static_cast<double>(matching) / static_cast<double>(probed);
}

}
