#include "gncpy/measurements/Exceptions.h"


namespace lager::gncpy::measurements {

BadParams::BadParams(char const* const message) noexcept
: std::runtime_error(message) {
    
}
} // namespace lager::gncpy::measurements
