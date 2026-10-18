#pragma once

#include <string_view>

namespace hopfilter {

/// Semantic version of the hopfilter core library, e.g. "0.1.0".
[[nodiscard]] std::string_view version() noexcept;

}
