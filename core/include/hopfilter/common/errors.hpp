#pragma once

#include <stdexcept>

namespace hopfilter {

/// A caller passed a value that violates an API contract (wrong dimension, k = 0, ...).
class InvalidArgumentError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

/// The object is not in a state that allows the operation (index full, empty index, ...).
class IndexStateError : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

/// A serialized index is truncated, corrupted, or from an unsupported format version.
class SerializationError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

}
