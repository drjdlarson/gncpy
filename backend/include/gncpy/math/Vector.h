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

    explicit Vector(std::initializer_list<T> list) 
    : Matrix<T>(list.size(), 1, std::vector(list)) {

    }

    explicit Vector(uint8_t nElements)
    : Matrix<T>(nElements, 1) {

    }

    T& operator() (uint8_t elem) {
        if(elem >= this->size()) {
            throw BadIndex("Indexing outside vector");
        }
        if(this->numRows() == 1){
            return Matrix<T>::operator()(static_cast<uint8_t>(0), elem);
        }
        return Matrix<T>::operator()(elem, static_cast<uint8_t>(0));
    }

    T operator() (uint8_t elem) const {
        if(elem >= this->size()) {
            throw BadIndex("Indexing outside vector");
        }
        if(this->numRows() == 1){
            return Matrix<T>::operator()(static_cast<uint8_t>(0), elem);
        }
        return Matrix<T>::operator()(elem, static_cast<uint8_t>(0));
    }

};
    
} // namespace lager::gncpy::matrix
