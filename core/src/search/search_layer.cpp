#include <algorithm>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include <hopfilter/common/types.hpp>
#include <hopfilter/search/search_context.hpp>
#include <hopfilter/search/search_layer.hpp>

namespace hopfilter {
namespace {

struct FartherFirst {
    [[nodiscard]] bool operator()(const Neighbor& first, const Neighbor& second) const noexcept {
        return CloserFirst{}(second, first);
    }
};

class LayerSearch {
public:
    LayerSearch(const SearchContext& context, std::span<const float> query, std::size_t ef)
        : context_(&context), query_(query), ef_(ef) {
        candidates_.reserve(ef);
        results_.reserve(ef + 1);
    }

    void seed(std::span<const Neighbor> entryPoints) {
        for (const Neighbor& entry : entryPoints) {
            if (context_->visited().markVisited(entry.id)) {
                offer(entry);
            }
        }
    }

    void run(LevelIndex level) {
        while (!candidates_.empty()) {
            std::ranges::pop_heap(candidates_, FartherFirst{});
            const Neighbor closest = candidates_.back();
            candidates_.pop_back();
            if (isFull() && worseThanWorstResult(closest.distance)) {
                return;
            }
            expand(closest.id, level);
        }
    }

    [[nodiscard]] SearchResults takeSorted() {
        std::ranges::sort(results_, CloserFirst{});
        return std::move(results_);
    }

private:
    void expand(NodeId node, LevelIndex level) {
        for (const NodeId neighbor : context_->graph().neighbors(node, level)) {
            if (!context_->visited().markVisited(neighbor)) {
                continue;
            }
            const float distance = context_->distances().toQuery(query_, neighbor);
            if (!isFull() || !worseThanWorstResult(distance)) {
                offer({neighbor, distance});
            }
        }
    }

    void offer(const Neighbor& neighbor) {
        candidates_.push_back(neighbor);
        std::ranges::push_heap(candidates_, FartherFirst{});
        results_.push_back(neighbor);
        std::ranges::push_heap(results_, CloserFirst{});
        if (results_.size() > ef_) {
            std::ranges::pop_heap(results_, CloserFirst{});
            results_.pop_back();
        }
    }

    [[nodiscard]] bool isFull() const noexcept { return results_.size() >= ef_; }

    [[nodiscard]] bool worseThanWorstResult(float distance) const noexcept {
        return distance > results_.front().distance;
    }

    const SearchContext* context_;
    std::span<const float> query_;
    std::size_t ef_;
    std::vector<Neighbor> candidates_;
    std::vector<Neighbor> results_;
};

}

SearchResults searchLayer(const SearchContext& context,
                          std::span<const float> query,
                          std::span<const Neighbor> entryPoints,
                          std::size_t ef,
                          LevelIndex level) {
    context.visited().clear();
    LayerSearch search{context, query, std::max<std::size_t>(ef, 1)};
    search.seed(entryPoints);
    search.run(level);
    return search.takeSorted();
}

Neighbor greedyDescend(const SearchContext& context,
                       std::span<const float> query,
                       Neighbor entry,
                       LevelIndex topLevel,
                       LevelIndex stopLevel) {
    Neighbor closest = entry;
    for (LevelIndex level = topLevel; level > stopLevel; --level) {
        closest = searchLayer(context, query, {&closest, 1}, 1, level).front();
    }
    return closest;
}

}
