#include "gncpy/dynamics/Exceptions.h"

namespace lager::gncpy::dynamics {

NoControlError::NoControlError() noexcept
: std::runtime_error("No control model set") {
    
}


NoStateConstraintError::NoStateConstraintError() noexcept
: std::runtime_error("No state constraint set") {
    
}
    
} // namespace lager::gncpy::dynamics
