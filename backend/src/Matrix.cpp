#include "Matrix.h"

namespace lager::gncpy::matrix {

BadIndex::BadIndex(char const* const message) throw()
: std::runtime_error(message) {
    
}
    
Matrix Matrix::operator+(const Matrix& m){
    if (!isSameSize(m)){
        throw BadDimension("Matrices dimensions are not equal")
    }
    for (uint8_t i = 0; i < numRows() * numCols(), i++){
        m_data[i] += m.m_data[i];
    }

} 

} // namespace lager::gncpy::matrix 