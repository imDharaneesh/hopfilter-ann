#include <hopfilter/common/version.hpp>

namespace hopfilter {

std::string_view version() noexcept {
    return HOPFILTER_VERSION;
}

}
