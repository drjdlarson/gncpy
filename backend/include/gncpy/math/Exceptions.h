#pragma once
#include <stdexcept>

namespace lager::gncpy::matrix {

class BadIndex final: public std::runtime_error {
public:
    explicit BadIndex(char const* const message) noexcept;
};


class BadDimension final: public std::runtime_error {
public:
    BadDimension() noexcept;
    explicit BadDimension(char const* const message) noexcept;
};


} // namespace lager::gncpy::matrix
