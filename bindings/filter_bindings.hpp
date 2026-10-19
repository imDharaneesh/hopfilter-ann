#pragma once

#include <pybind11/pybind11.h>

namespace hopfilter::bindings {

void bindFilter(pybind11::module_& module);

}
