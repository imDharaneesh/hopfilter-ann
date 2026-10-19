#pragma once

#include <stdexcept>

namespace hopfilter {

class InvalidArgumentError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

class IndexStateError : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

class SerializationError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

}
