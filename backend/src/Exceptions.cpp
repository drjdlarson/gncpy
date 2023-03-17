#include "gncpy/Exceptions.h"


namespace lager::gncpy::exceptions {

BadParams::BadParams(char const* const message) noexcept
: std::runtime_error(message) {
}


TypeError::TypeError(char const* const message) noexcept
: std::runtime_error(message) {
    
}
} // namespace lager::gncpy::measurements
