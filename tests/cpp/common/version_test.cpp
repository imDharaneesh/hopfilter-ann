#include <regex>
#include <string>

#include <gtest/gtest.h>

#include <hopfilter/common/version.hpp>

namespace hopfilter {
namespace {

TEST(VersionTest, MatchesProjectVersion) {
    EXPECT_EQ(version(), HOPFILTER_EXPECTED_VERSION);
}

TEST(VersionTest, IsSemanticVersion) {
    const std::regex semverPattern{R"(^\d+\.\d+\.\d+$)"};
    EXPECT_TRUE(std::regex_match(std::string{version()}, semverPattern));
}

}  // namespace
}  // namespace hopfilter
