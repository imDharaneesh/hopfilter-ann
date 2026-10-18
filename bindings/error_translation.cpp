#include "error_translation.hpp"

#include <exception>

#include <pybind11/pybind11.h>

#include <hopfilter/common/errors.hpp>

namespace hopfilter::bindings {
namespace {

void raise(const char* errorName, const std::exception& error) {
    const auto errors = pybind11::module_::import("hopfilter.errors");
    PyErr_SetString(errors.attr(errorName).ptr(), error.what());
}

void translate(std::exception_ptr pending) {
    try {
        if (pending) {
            std::rethrow_exception(pending);
        }
    } catch (const InvalidArgumentError& error) {
        raise("InvalidArgumentError", error);
    } catch (const IndexStateError& error) {
        raise("IndexStateError", error);
    } catch (const SerializationError& error) {
        raise("SerializationError", error);
    }
}

}  // namespace

void registerErrorTranslation() {
    pybind11::register_exception_translator(&translate);
}

}  // namespace hopfilter::bindings
