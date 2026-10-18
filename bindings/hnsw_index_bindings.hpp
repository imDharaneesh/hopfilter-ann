#pragma once

#include <pybind11/pybind11.h>

namespace hopfilter::bindings {

/// Adds the ``HnswIndex`` class to ``module``.
void bindHnswIndex(pybind11::module_& module);

}  // namespace hopfilter::bindings
