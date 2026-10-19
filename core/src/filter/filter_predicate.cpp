#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/common/types.hpp>
#include <hopfilter/filter/filter_predicate.hpp>

namespace hopfilter {
namespace {

std::vector<PredicatePtr> requireChildren(std::vector<PredicatePtr> children) {
    if (children.empty()) {
        throw InvalidArgumentError("a combined filter needs at least one child");
    }
    if (std::ranges::any_of(children, [](const PredicatePtr& child) { return !child; })) {
        throw InvalidArgumentError("a combined filter cannot have an empty child");
    }
    return children;
}

}

void FilterPredicate::collectMatches(std::size_t universeSize, std::vector<NodeId>& matches) const {
    for (std::size_t id = 0; id < universeSize; ++id) {
        if (this->matches(static_cast<NodeId>(id))) {
            matches.push_back(static_cast<NodeId>(id));
        }
    }
}

NumericRange::NumericRange(const NumericColumn& column, double low, double high)
    : column_(&column), low_(low), high_(high) {
    if (std::isnan(low) || std::isnan(high) || low > high) {
        throw InvalidArgumentError("numeric range needs low <= high");
    }
}

void BitsetPredicate::collectMatches(std::size_t universeSize, std::vector<NodeId>& matches) const {
    bits_.forEachSet([&](std::size_t position) {
        if (position < universeSize) {
            matches.push_back(static_cast<NodeId>(position));
        }
    });
}

AndPredicate::AndPredicate(std::vector<PredicatePtr> children)
    : children_(requireChildren(std::move(children))) {}

bool AndPredicate::matches(NodeId id) const noexcept {
    return std::ranges::all_of(children_,
                               [id](const PredicatePtr& child) { return child->matches(id); });
}

OrPredicate::OrPredicate(std::vector<PredicatePtr> children)
    : children_(requireChildren(std::move(children))) {}

bool OrPredicate::matches(NodeId id) const noexcept {
    return std::ranges::any_of(children_,
                               [id](const PredicatePtr& child) { return child->matches(id); });
}

NotPredicate::NotPredicate(PredicatePtr child) : child_(std::move(child)) {
    if (!child_) {
        throw InvalidArgumentError("a negated filter cannot be empty");
    }
}

}
