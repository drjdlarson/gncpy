#pragma once
#include <stdexcept>

namespace lager::gncpy::exceptions {

class BadParams final: public std::runtime_error {
public:
    explicit BadParams(char const* const message) noexcept;
};

class TypeError final: public std::runtime_error {
public:
    explicit TypeError(char const* const message) noexcept;
};

} // namespace lager::gncpy::exceptions