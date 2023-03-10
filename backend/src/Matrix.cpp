#include "Matrix.h"

namespace lager::gncpy::matrix {

BadIndex::BadIndex(char const* const message) noexcept
: std::runtime_error(message) {
    
}
    
} // namespace lager::gncpy::matrix 