#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/filter/dynamic_bitset.hpp>

namespace hopfilter {
namespace {

TEST(DynamicBitsetTest, StartsEmpty) {
    const DynamicBitset bits{130};

    EXPECT_EQ(bits.size(), 130U);
    EXPECT_EQ(bits.popcount(), 0U);
    EXPECT_FALSE(bits.test(0));
    EXPECT_FALSE(bits.test(129));
}

TEST(DynamicBitsetTest, SetsBitsAcrossWordBoundaries) {
    DynamicBitset bits{130};
    for (const std::size_t position : {0U, 63U, 64U, 129U}) {
        bits.set(position);
    }

    EXPECT_TRUE(bits.test(63));
    EXPECT_TRUE(bits.test(64));
    EXPECT_FALSE(bits.test(65));
    EXPECT_EQ(bits.popcount(), 4U);
}

TEST(DynamicBitsetTest, IgnoresPositionsPastTheEnd) {
    DynamicBitset bits{10};
    bits.set(10);
    bits.set(1000);

    EXPECT_EQ(bits.popcount(), 0U);
    EXPECT_FALSE(bits.test(10));
    EXPECT_FALSE(bits.test(1000));
}

TEST(DynamicBitsetTest, VisitsSetBitsInAscendingOrder) {
    DynamicBitset bits{200};
    const std::vector<std::size_t> expected{3, 64, 65, 127, 199};
    for (const std::size_t position : expected) {
        bits.set(position);
    }

    std::vector<std::size_t> visited;
    bits.forEachSet([&visited](std::size_t position) { visited.push_back(position); });

    EXPECT_EQ(visited, expected);
}

TEST(DynamicBitsetTest, DefaultConstructedIsEmpty) {
    const DynamicBitset bits;

    EXPECT_EQ(bits.size(), 0U);
    EXPECT_FALSE(bits.test(0));
}

}
}
