#include "gncpy/math/Exceptions.h"


namespace lager::gncpy::matrix {

BadIndex::BadIndex(char const* const message) noexcept
: std::runtime_error(message) {
    
}


BadDimension::BadDimension() noexcept
: std::runtime_error("Matrix dimensions are not equal") {
    
}


BadDimension::BadDimension(char const* const message) noexcept
: std::runtime_error(message) {

}

} // namespace lager::gncpy::matrix
