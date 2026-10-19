#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <hopfilter/common/types.hpp>
#include <hopfilter/filter/attribute_store.hpp>
#include <hopfilter/filter/dynamic_bitset.hpp>

namespace hopfilter {

class FilterPredicate {
public:
    virtual ~FilterPredicate() = default;

    [[nodiscard]] virtual bool matches(NodeId id) const noexcept = 0;

    virtual void collectMatches(std::size_t universeSize, std::vector<NodeId>& matches) const;
};

using PredicatePtr = std::unique_ptr<FilterPredicate>;

class CategoryEquals final : public FilterPredicate {
public:
    CategoryEquals(const CategoricalColumn& column, CategoryCode value) noexcept
        : column_(&column), value_(value) {}

    [[nodiscard]] bool matches(NodeId id) const noexcept override {
        return id < column_->size() && (*column_)[id] == value_;
    }

private:
    const CategoricalColumn* column_;
    CategoryCode value_;
};

class NumericRange final : public FilterPredicate {
public:
    NumericRange(const NumericColumn& column, double low, double high);

    [[nodiscard]] bool matches(NodeId id) const noexcept override {
        if (id >= column_->size()) {
            return false;
        }
        const double value = (*column_)[id];
        return value >= low_ && value <= high_;
    }

private:
    const NumericColumn* column_;
    double low_;
    double high_;
};

class BitsetPredicate final : public FilterPredicate {
public:
    explicit BitsetPredicate(DynamicBitset bits) noexcept : bits_(std::move(bits)) {}

    [[nodiscard]] bool matches(NodeId id) const noexcept override { return bits_.test(id); }

    void collectMatches(std::size_t universeSize, std::vector<NodeId>& matches) const override;

    [[nodiscard]] std::size_t popcount() const noexcept { return bits_.popcount(); }
    [[nodiscard]] const DynamicBitset& bits() const noexcept { return bits_; }

private:
    DynamicBitset bits_;
};

class AndPredicate final : public FilterPredicate {
public:
    explicit AndPredicate(std::vector<PredicatePtr> children);

    [[nodiscard]] bool matches(NodeId id) const noexcept override;

private:
    std::vector<PredicatePtr> children_;
};

class OrPredicate final : public FilterPredicate {
public:
    explicit OrPredicate(std::vector<PredicatePtr> children);

    [[nodiscard]] bool matches(NodeId id) const noexcept override;

private:
    std::vector<PredicatePtr> children_;
};

class NotPredicate final : public FilterPredicate {
public:
    explicit NotPredicate(PredicatePtr child);

    [[nodiscard]] bool matches(NodeId id) const noexcept override { return !child_->matches(id); }

private:
    PredicatePtr child_;
};

}
