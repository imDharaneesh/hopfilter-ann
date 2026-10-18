#pragma once

namespace hopfilter::bindings {

/// Maps hopfilter's C++ exceptions onto the matching classes in ``hopfilter.errors``.
void registerErrorTranslation();

}  // namespace hopfilter::bindings
