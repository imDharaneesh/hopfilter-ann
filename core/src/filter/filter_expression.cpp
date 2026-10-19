#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/filter/attribute_store.hpp>
#include <hopfilter/filter/dynamic_bitset.hpp>
#include <hopfilter/filter/filter_expression.hpp>
#include <hopfilter/filter/filter_predicate.hpp>

namespace hopfilter {
namespace {

struct Equals {
    std::string column;
    CategoryCode value;
};

struct Between {
    std::string column;
    double low;
    double high;
};

struct Mask {
    DynamicBitset bits;
};

struct AllOf {
    std::vector<FilterExpression> children;
};

struct AnyOf {
    std::vector<FilterExpression> children;
};

struct Negation {
    FilterExpression child;
};

template <typename Kind>
constexpr bool kIsCombination = std::is_same_v<Kind, AllOf> || std::is_same_v<Kind, AnyOf>;

void requireColumnName(const std::string& column) {
    if (column.empty()) {
        throw InvalidArgumentError("filter column name must be non-empty");
    }
}

std::string joined(const std::vector<FilterExpression>& children, const char* separator) {
    std::string text = "(";
    for (std::size_t index = 0; index < children.size(); ++index) {
        text += (index == 0 ? "" : separator) + children[index].describe();
    }
    return text + ")";
}

}

struct FilterExpression::Node {
    std::variant<Equals, Between, Mask, AllOf, AnyOf, Negation> value;
};

namespace {

template <typename Kind>
std::vector<FilterExpression> flattened(std::vector<FilterExpression> children,
                                        const std::vector<const Kind*>& nested) {
    std::vector<FilterExpression> result;
    for (std::size_t index = 0; index < children.size(); ++index) {
        if (nested[index] == nullptr) {
            result.push_back(std::move(children[index]));
            continue;
        }
        result.insert(result.end(), nested[index]->children.begin(), nested[index]->children.end());
    }
    return result;
}

std::vector<PredicatePtr> compileAll(const std::vector<FilterExpression>& children,
                                     const AttributeStore& attributes) {
    std::vector<PredicatePtr> predicates;
    predicates.reserve(children.size());
    for (const FilterExpression& child : children) {
        predicates.push_back(child.compile(attributes));
    }
    return predicates;
}

std::string numberText(double value) {
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

}

FilterExpression FilterExpression::equals(std::string column, CategoryCode value) {
    requireColumnName(column);
    return FilterExpression{std::make_shared<const Node>(Node{Equals{std::move(column), value}})};
}

FilterExpression FilterExpression::between(std::string column, double low, double high) {
    requireColumnName(column);
    if (std::isnan(low) || std::isnan(high) || low > high) {
        throw InvalidArgumentError("filter range on '" + column + "' needs low <= high");
    }
    return FilterExpression{
        std::make_shared<const Node>(Node{Between{std::move(column), low, high}})};
}

FilterExpression FilterExpression::mask(DynamicBitset bits) {
    return FilterExpression{std::make_shared<const Node>(Node{Mask{std::move(bits)}})};
}

FilterExpression FilterExpression::allOf(std::vector<FilterExpression> children) {
    if (children.empty()) {
        throw InvalidArgumentError("allOf needs at least one filter");
    }
    std::vector<const AllOf*> nested;
    nested.reserve(children.size());
    for (const FilterExpression& child : children) {
        nested.push_back(std::get_if<AllOf>(&child.node_->value));
    }
    auto flat = flattened(std::move(children), nested);
    return FilterExpression{std::make_shared<const Node>(Node{AllOf{std::move(flat)}})};
}

FilterExpression FilterExpression::anyOf(std::vector<FilterExpression> children) {
    if (children.empty()) {
        throw InvalidArgumentError("anyOf needs at least one filter");
    }
    std::vector<const AnyOf*> nested;
    nested.reserve(children.size());
    for (const FilterExpression& child : children) {
        nested.push_back(std::get_if<AnyOf>(&child.node_->value));
    }
    auto flat = flattened(std::move(children), nested);
    return FilterExpression{std::make_shared<const Node>(Node{AnyOf{std::move(flat)}})};
}

FilterExpression FilterExpression::negation(FilterExpression child) {
    return FilterExpression{std::make_shared<const Node>(Node{Negation{std::move(child)}})};
}

PredicatePtr FilterExpression::compile(const AttributeStore& attributes) const {
    return std::visit(
        [&attributes](const auto& kind) -> PredicatePtr {
            using Kind = std::decay_t<decltype(kind)>;
            if constexpr (std::is_same_v<Kind, Equals>) {
                return std::make_unique<CategoryEquals>(attributes.categorical(kind.column),
                                                        kind.value);
            } else if constexpr (std::is_same_v<Kind, Between>) {
                return std::make_unique<NumericRange>(
                    attributes.numeric(kind.column), kind.low, kind.high);
            } else if constexpr (std::is_same_v<Kind, Mask>) {
                return std::make_unique<BitsetPredicate>(kind.bits);
            } else if constexpr (std::is_same_v<Kind, AllOf>) {
                return std::make_unique<AndPredicate>(compileAll(kind.children, attributes));
            } else if constexpr (std::is_same_v<Kind, AnyOf>) {
                return std::make_unique<OrPredicate>(compileAll(kind.children, attributes));
            } else {
                return std::make_unique<NotPredicate>(kind.child.compile(attributes));
            }
        },
        node_->value);
}

std::string FilterExpression::describe() const {
    return std::visit(
        [](const auto& kind) -> std::string {
            using Kind = std::decay_t<decltype(kind)>;
            if constexpr (std::is_same_v<Kind, Equals>) {
                return kind.column + " == " + std::to_string(kind.value);
            } else if constexpr (std::is_same_v<Kind, Between>) {
                return kind.column + " in [" + numberText(kind.low) + ", " + numberText(kind.high) +
                       "]";
            } else if constexpr (std::is_same_v<Kind, Mask>) {
                return "mask(" + std::to_string(kind.bits.popcount()) + " of " +
                       std::to_string(kind.bits.size()) + ")";
            } else if constexpr (kIsCombination<Kind>) {
                return joined(kind.children, std::is_same_v<Kind, AllOf> ? " & " : " | ");
            } else {
                return "~" + kind.child.describe();
            }
        },
        node_->value);
}

}
