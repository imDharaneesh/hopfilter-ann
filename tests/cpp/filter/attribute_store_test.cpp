#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hopfilter/common/errors.hpp>
#include <hopfilter/filter/attribute_store.hpp>

namespace hopfilter {
namespace {

TEST(AttributeStoreTest, StartsEmpty) {
    const AttributeStore store;

    EXPECT_TRUE(store.empty());
    EXPECT_TRUE(store.names().empty());
}

TEST(AttributeStoreTest, StoresColumnsByKind) {
    AttributeStore store;
    store.setCategorical("genre", {1, 2, 1});
    store.setNumeric("year", {1990.0, 2001.0, 1985.0});

    EXPECT_EQ(store.categorical("genre"), (CategoricalColumn{1, 2, 1}));
    EXPECT_EQ(store.numeric("year"), (NumericColumn{1990.0, 2001.0, 1985.0}));
    EXPECT_EQ(store.names(), (std::vector<std::string>{"genre", "year"}));
}

TEST(AttributeStoreTest, ReplacingANameChangesItsKind) {
    AttributeStore store;
    store.setCategorical("price", {1, 2});
    store.setNumeric("price", {9.5, 3.0});

    EXPECT_EQ(store.numeric("price"), (NumericColumn{9.5, 3.0}));
    EXPECT_THROW(static_cast<void>(store.categorical("price")), InvalidArgumentError);
    EXPECT_EQ(store.names().size(), 1U);
}

TEST(AttributeStoreTest, RejectsUnknownAndMistypedNames) {
    AttributeStore store;
    store.setCategorical("genre", {1});

    EXPECT_THROW(static_cast<void>(store.numeric("genre")), InvalidArgumentError);
    EXPECT_THROW(static_cast<void>(store.categorical("missing")), InvalidArgumentError);
}

TEST(AttributeStoreTest, RejectsEmptyNames) {
    AttributeStore store;

    EXPECT_THROW(store.setCategorical("", {}), InvalidArgumentError);
    EXPECT_THROW(store.setNumeric("", {}), InvalidArgumentError);
}

}
}
