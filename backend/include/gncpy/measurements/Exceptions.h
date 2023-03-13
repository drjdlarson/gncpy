#pragma once
#include <stdexcept>

namespace lager::gncpy::measurements {

class BadParams final: public std::runtime_error {
public:
    explicit BadParams(char const* const message) noexcept;
};


} // namespace lager::gncpy::measurements