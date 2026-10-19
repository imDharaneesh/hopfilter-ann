#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <hopfilter/filter/attribute_store.hpp>
#include <hopfilter/filter/dynamic_bitset.hpp>
#include <hopfilter/filter/filter_predicate.hpp>

namespace hopfilter {

class FilterExpression {
public:
    [[nodiscard]] static FilterExpression equals(std::string column, CategoryCode value);
    [[nodiscard]] static FilterExpression between(std::string column, double low, double high);
    [[nodiscard]] static FilterExpression mask(DynamicBitset bits);
    [[nodiscard]] static FilterExpression allOf(std::vector<FilterExpression> children);
    [[nodiscard]] static FilterExpression anyOf(std::vector<FilterExpression> children);
    [[nodiscard]] static FilterExpression negation(FilterExpression child);

    [[nodiscard]] PredicatePtr compile(const AttributeStore& attributes) const;

    [[nodiscard]] std::string describe() const;

    friend FilterExpression operator&(const FilterExpression& lhs, const FilterExpression& rhs) {
        return allOf({lhs, rhs});
    }
    friend FilterExpression operator|(const FilterExpression& lhs, const FilterExpression& rhs) {
        return anyOf({lhs, rhs});
    }
    friend FilterExpression operator!(const FilterExpression& expression) {
        return negation(expression);
    }

    struct Node;

private:
    explicit FilterExpression(std::shared_ptr<const Node> node) noexcept : node_(std::move(node)) {}

    std::shared_ptr<const Node> node_;
};

}
