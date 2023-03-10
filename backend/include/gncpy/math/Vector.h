#pragma once
#include <algorithm>
#include "gncpy/math/Matrix.h"
#include "gncpy/math/Exceptions.h"


namespace lager::gncpy::matrix {

template<typename T>
class Vector final: public Matrix<T> {

public:
    Vector(uint8_t nElements, std::vector<T> data) 
    : Matrix<T>(nElements, 1, data) {
        
    }

    explicit Vector(std::initializer_list<std::initializer_list<T>> listlist) {
        uint8_t nRows = listlist.begin()->size();
        uint8_t nCols = listlist.size();
        if(nRows == 1) {
            Matrix<T>(nCols, 1);
        } else if(nCols == 1) {
            Matrix<T>(nRows, 1);
        } else if(nRows == 0 || nCols == 0) {
            throw BadDimension("Vector can not have size 0");
        }
        else {
            throw BadDimension("Vector must have at least 1 dimension = 1");
        }
    }

    explicit Vector(std::initializer_list<T> list) 
    : Matrix<T>(list.size(), 1, std::vector(list)) {

    }

    explicit Vector(uint8_t nElements)
    : Matrix<T>(nElements, 1) {

    }

};
    
} // namespace lager::gncpy::matrix
