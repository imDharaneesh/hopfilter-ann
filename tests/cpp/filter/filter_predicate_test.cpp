#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/common/types.hpp>
#include <hopfilter/filter/attribute_store.hpp>
#include <hopfilter/filter/dynamic_bitset.hpp>
#include <hopfilter/filter/filter_predicate.hpp>

namespace hopfilter {
namespace {

std::vector<NodeId> matchesOf(const FilterPredicate& predicate, std::size_t universeSize) {
    std::vector<NodeId> matches;
    predicate.collectMatches(universeSize, matches);
    return matches;
}

std::vector<PredicatePtr> pair(PredicatePtr first, PredicatePtr second) {
    std::vector<PredicatePtr> children;
    children.push_back(std::move(first));
    children.push_back(std::move(second));
    return children;
}

class FilterPredicateTest : public ::testing::Test {
protected:
    [[nodiscard]] PredicatePtr genre(CategoryCode value) const {
        return std::make_unique<CategoryEquals>(genres_, value);
    }

    [[nodiscard]] PredicatePtr year(double low, double high) const {
        return std::make_unique<NumericRange>(years_, low, high);
    }

private:
    CategoricalColumn genres_{0, 1, 1, 2, 1};
    NumericColumn years_{1990.0, 1995.0, 2000.0, 2005.0, 2010.0};
};

TEST_F(FilterPredicateTest, CategoryEqualsMatchesItsCode) {
    EXPECT_EQ(matchesOf(*genre(1), 5), (std::vector<NodeId>{1, 2, 4}));
    EXPECT_TRUE(matchesOf(*genre(7), 5).empty());
}

TEST_F(FilterPredicateTest, NumericRangeIsClosedOnBothEnds) {
    EXPECT_EQ(matchesOf(*year(1995.0, 2005.0), 5), (std::vector<NodeId>{1, 2, 3}));
}

TEST_F(FilterPredicateTest, NumericRangeRejectsInvertedBounds) {
    const NumericColumn column{1.0};

    EXPECT_THROW(NumericRange(column, 2.0, 1.0), InvalidArgumentError);
}

TEST_F(FilterPredicateTest, RowsPastTheColumnNeverMatch) {
    EXPECT_FALSE(genre(1)->matches(5));
    EXPECT_FALSE(year(0.0, 3000.0)->matches(100));
    EXPECT_EQ(matchesOf(*genre(1), 8), (std::vector<NodeId>{1, 2, 4}));
}

TEST_F(FilterPredicateTest, CombinesWithAndOrNot) {
    const AndPredicate both{pair(genre(1), year(1995.0, 2000.0))};
    const OrPredicate either{pair(genre(0), genre(2))};
    const NotPredicate notOne{genre(1)};

    EXPECT_EQ(matchesOf(both, 5), (std::vector<NodeId>{1, 2}));
    EXPECT_EQ(matchesOf(either, 5), (std::vector<NodeId>{0, 3}));
    EXPECT_EQ(matchesOf(notOne, 5), (std::vector<NodeId>{0, 3}));
}

TEST_F(FilterPredicateTest, CombinationsRejectMissingChildren) {
    EXPECT_THROW(AndPredicate{std::vector<PredicatePtr>{}}, InvalidArgumentError);
    EXPECT_THROW(OrPredicate{pair(genre(1), nullptr)}, InvalidArgumentError);
    EXPECT_THROW(NotPredicate{nullptr}, InvalidArgumentError);
}

TEST(BitsetPredicateTest, CollectsOnlyIdsInsideTheUniverse) {
    DynamicBitset bits{100};
    for (const NodeId id : {2U, 40U, 70U, 99U}) {
        bits.set(id);
    }
    const BitsetPredicate predicate{bits};

    EXPECT_EQ(predicate.popcount(), 4U);
    EXPECT_TRUE(predicate.matches(40));
    EXPECT_FALSE(predicate.matches(41));
    EXPECT_EQ(matchesOf(predicate, 71), (std::vector<NodeId>{2, 40, 70}));
}

}
}
