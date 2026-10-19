#include <cstddef>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/common/types.hpp>
#include <hopfilter/filter/attribute_store.hpp>
#include <hopfilter/filter/dynamic_bitset.hpp>
#include <hopfilter/filter/filter_expression.hpp>

namespace hopfilter {
namespace {

constexpr std::size_t kRows = 6;

class FilterExpressionTest : public ::testing::Test {
protected:
    FilterExpressionTest() {
        store_.setCategorical("genre", {0, 1, 1, 2, 1, 0});
        store_.setNumeric("year", {1990.0, 1995.0, 2000.0, 2005.0, 2010.0, 2015.0});
    }

    [[nodiscard]] std::vector<NodeId> matchesOf(const FilterExpression& expression) const {
        std::vector<NodeId> matches;
        expression.compile(store_)->collectMatches(kRows, matches);
        return matches;
    }

    [[nodiscard]] const AttributeStore& store() const { return store_; }

private:
    AttributeStore store_;
};

TEST_F(FilterExpressionTest, CompilesLeaves) {
    EXPECT_EQ(matchesOf(FilterExpression::equals("genre", 1)), (std::vector<NodeId>{1, 2, 4}));
    EXPECT_EQ(matchesOf(FilterExpression::between("year", 2000.0, 2010.0)),
              (std::vector<NodeId>{2, 3, 4}));
}

TEST_F(FilterExpressionTest, CompilesOperators) {
    const auto genreOne = FilterExpression::equals("genre", 1);
    const auto recent = FilterExpression::between("year", 2000.0, 3000.0);

    EXPECT_EQ(matchesOf(genreOne & recent), (std::vector<NodeId>{2, 4}));
    EXPECT_EQ(matchesOf(genreOne | recent), (std::vector<NodeId>{1, 2, 3, 4, 5}));
    EXPECT_EQ(matchesOf(!genreOne), (std::vector<NodeId>{0, 3, 5}));
}

TEST_F(FilterExpressionTest, CompilesMasks) {
    DynamicBitset bits{kRows};
    bits.set(0);
    bits.set(4);

    EXPECT_EQ(matchesOf(FilterExpression::mask(bits)), (std::vector<NodeId>{0, 4}));
}

TEST_F(FilterExpressionTest, FlattensChainedCombinations) {
    const auto chained = FilterExpression::equals("genre", 0) &
                         FilterExpression::equals("genre", 1) &
                         FilterExpression::equals("genre", 2);

    EXPECT_EQ(chained.describe(), "(genre == 0 & genre == 1 & genre == 2)");
}

TEST_F(FilterExpressionTest, DescribesEveryKind) {
    const auto expression = !(FilterExpression::between("year", 1990.5, 2000.0) |
                              FilterExpression::mask(DynamicBitset{4}));

    EXPECT_EQ(expression.describe(), "~(year in [1990.5, 2000] | mask(0 of 4))");
}

TEST_F(FilterExpressionTest, RejectsInvalidLeavesEagerly) {
    EXPECT_THROW(static_cast<void>(FilterExpression::equals("", 1)), InvalidArgumentError);
    EXPECT_THROW(static_cast<void>(FilterExpression::between("year", 2.0, 1.0)),
                 InvalidArgumentError);
    EXPECT_THROW(static_cast<void>(FilterExpression::between(
                     "year", std::numeric_limits<double>::quiet_NaN(), 1.0)),
                 InvalidArgumentError);
    EXPECT_THROW(static_cast<void>(FilterExpression::allOf({})), InvalidArgumentError);
    EXPECT_THROW(static_cast<void>(FilterExpression::anyOf({})), InvalidArgumentError);
}

TEST_F(FilterExpressionTest, CompilingChecksColumnsAgainstTheStore) {
    EXPECT_THROW(static_cast<void>(FilterExpression::equals("missing", 1).compile(store())),
                 InvalidArgumentError);
    EXPECT_THROW(static_cast<void>(FilterExpression::equals("year", 1).compile(store())),
                 InvalidArgumentError);
    EXPECT_THROW(static_cast<void>(FilterExpression::between("genre", 0.0, 1.0).compile(store())),
                 InvalidArgumentError);
}

}
}
