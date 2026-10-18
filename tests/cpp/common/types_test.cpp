#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/common/types.hpp>

namespace hopfilter {
namespace {

TEST(CloserFirstTest, OrdersByDistanceThenId) {
    std::vector<Neighbor> neighbors{{7, 0.5F}, {3, 0.25F}, {2, 0.5F}, {9, 0.1F}};

    std::ranges::sort(neighbors, CloserFirst{});

    const std::vector<Neighbor> expected{{9, 0.1F}, {3, 0.25F}, {2, 0.5F}, {7, 0.5F}};
    EXPECT_EQ(neighbors, expected);
}

TEST(CloserFirstTest, IsIrreflexive) {
    const Neighbor neighbor{4, 1.0F};

    EXPECT_FALSE(CloserFirst{}(neighbor, neighbor));
}

TEST(ErrorsTest, DeriveFromMatchingStandardExceptions) {
    EXPECT_TRUE((std::is_base_of_v<std::invalid_argument, InvalidArgumentError>));
    EXPECT_TRUE((std::is_base_of_v<std::logic_error, IndexStateError>));
    EXPECT_TRUE((std::is_base_of_v<std::runtime_error, SerializationError>));
}

}
}
