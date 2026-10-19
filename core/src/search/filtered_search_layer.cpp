#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include <hopfilter/common/types.hpp>
#include <hopfilter/filter/filter_predicate.hpp>
#include <hopfilter/search/search_context.hpp>
#include <hopfilter/search/search_layer.hpp>

namespace hopfilter {
namespace {

class FilteredLayerSearch {
public:
    FilteredLayerSearch(const SearchContext& context,
                        std::span<const float> query,
                        const FilterPredicate& filter,
                        const EfSchedule& schedule)
        : context_(&context),
          query_(query),
          filter_(&filter),
          schedule_(schedule),
          ef_(std::max<std::size_t>(schedule.ef, 1)) {
        results_.reserve(ef_ + 1);
    }

    void seed(std::span<const Neighbor> entryPoints) {
        for (const Neighbor& entry : entryPoints) {
            if (context_->visited().markVisited(entry.id)) {
                consider(entry, filter_->matches(entry.id));
            }
        }
    }

    void run() {
        while (!candidates_.empty()) {
            std::ranges::pop_heap(candidates_, FartherFirst{});
            const Neighbor closest = candidates_.back();
            candidates_.pop_back();
            if (beyondWorstResult(closest.distance)) {
                return;
            }
            if (!candidates_.empty()) {
                context_->graph().prefetchNeighbors(candidates_.front().id, 0);
            }
            expand(closest.id);
        }
    }

    [[nodiscard]] SearchResults takeSorted() {
        std::ranges::sort(results_, CloserFirst{});
        return std::move(results_);
    }

private:
    void expand(NodeId node) {
        const auto neighbors = context_->graph().neighbors(node, 0);
        for (std::size_t position = 0; position < neighbors.size(); ++position) {
            if (position + 1 < neighbors.size()) {
                context_->visited().prefetch(neighbors[position + 1]);
                context_->distances().prefetch(neighbors[position + 1]);
            }
            visit(neighbors[position]);
        }
    }

    void visit(NodeId neighbor) {
        if (!context_->visited().markVisited(neighbor)) {
            return;
        }
        const float distance = context_->distances().toQuery(query_, neighbor);
        const bool matches = filter_->matches(neighbor);
        recordVisit(matches);
        if (!beyondWorstResult(distance)) {
            consider({neighbor, distance}, matches);
        }
    }

    void consider(const Neighbor& neighbor, bool matches) {
        candidates_.push_back(neighbor);
        std::ranges::push_heap(candidates_, FartherFirst{});
        if (!matches) {
            return;
        }
        results_.push_back(neighbor);
        std::ranges::push_heap(results_, CloserFirst{});
        if (results_.size() > ef_) {
            std::ranges::pop_heap(results_, CloserFirst{});
            results_.pop_back();
        }
    }

    void recordVisit(bool matches) noexcept {
        if (schedule_.growthFactor <= 1.0) {
            return;
        }
        ++windowVisits_;
        windowRejections_ += matches ? 0 : 1;
        if (windowVisits_ < schedule_.rejectionWindow) {
            return;
        }
        const double rejectionRate =
            static_cast<double>(windowRejections_) / static_cast<double>(windowVisits_);
        if (rejectionRate > schedule_.rejectionThreshold && ef_ < schedule_.maxEf) {
            const auto grown = static_cast<std::size_t>(
                std::ceil(static_cast<double>(ef_) * schedule_.growthFactor));
            ef_ = std::min(schedule_.maxEf, std::max(grown, ef_ + 1));
        }
        windowVisits_ = 0;
        windowRejections_ = 0;
    }

    [[nodiscard]] bool beyondWorstResult(float distance) const noexcept {
        return results_.size() >= ef_ && distance > results_.front().distance;
    }

    const SearchContext* context_;
    std::span<const float> query_;
    const FilterPredicate* filter_;
    EfSchedule schedule_;
    std::size_t ef_;
    std::size_t windowVisits_ = 0;
    std::size_t windowRejections_ = 0;
    std::vector<Neighbor> candidates_;
    std::vector<Neighbor> results_;
};

}

SearchResults filteredSearchLayer(const SearchContext& context,
                                  std::span<const float> query,
                                  std::span<const Neighbor> entryPoints,
                                  const FilterPredicate& filter,
                                  const EfSchedule& schedule) {
    context.visited().clear();
    FilteredLayerSearch search{context, query, filter, schedule};
    search.seed(entryPoints);
    search.run();
    return search.takeSorted();
}

}
