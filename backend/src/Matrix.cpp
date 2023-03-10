#include "Matrix.h"

namespace lager::gncpy::matrix {

BadIndex::BadIndex(char const* const message) throw()
: std::runtime_error(message) {
    
}
    
} // namespace lager::gncpy::matrix 