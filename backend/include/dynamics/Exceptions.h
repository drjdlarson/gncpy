#pragma once
#include <stdexcept>


namespace lager::gncpy::dynamics {

class NoControlError final: public std::runtime_error {
public:
    explicit NoControlError() noexcept;
};


class NoStateConstraintError final: public std::runtime_error {
public:
    explicit NoStateConstraintError() noexcept;
};
    
} // namespace lager::gncpy::dynamics
