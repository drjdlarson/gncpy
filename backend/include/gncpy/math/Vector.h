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

    explicit Vector(std::initializer_list<std::initializer_list<T>> listlist)
    : Matrix<T>(std::min(listlist.begin()->size(), listlist.size()) > 1 ? throw BadDimension("Invalid dimensions for vector") : listlist) {
        // uint8_t nRows = listlist.begin()->size();
        // uint8_t nCols = listlist.size();
        // if(nRows == 1) {
        //     Matrix<T>(nCols, 1);
        // } else if(nCols == 1) {
        //     Matrix<T>(nRows, 1);
        // } else if(nRows == 0 || nCols == 0) {
        //     throw BadDimension("Vector can not have size 0");
        // }
        // else {
        //     throw BadDimension("Vector must have at least 1 dimension = 1");
        // }

    }

    explicit Vector(std::initializer_list<T> list) 
    : Matrix<T>(list.size(), 1, std::vector(list)) {

    }

    explicit Vector(uint8_t nElements)
    : Matrix<T>(nElements, 1) {

    }

    // TODO: does this work correctly?
    T& operator() (uint8_t elem) {
        if(elem >= this->size()) {
            throw BadIndex("Indexing outside vector");
        }
        if(this->numRows() == 1){
            return Matrix<T>::operator()(static_cast<uint8_t>(1), elem);
            // return this->template operator()<T>(static_cast<uint8_t>(1), elem);
        }
        return Matrix<T>::operator()(elem, static_cast<uint8_t>(1));
        // return this->template operator()<T>(elem, static_cast<uint8_t>(1));
    }

    // TODO: does this work correctly?
    T operator() (uint8_t elem) const {
        if(elem >= this->size()) {
            throw BadIndex("Indexing outside vector");
        }
        if(this->numRows() == 1){
            return Matrix<T>::operator()(static_cast<uint8_t>(1), elem);
            // return this->template operator()<T>(static_cast<uint8_t>(1), elem);
        }
        return Matrix<T>::operator()(elem, static_cast<uint8_t>(1));
        // return this->template operator()<T>(elem, static_cast<uint8_t>(1));
    }

};
    
} // namespace lager::gncpy::matrix
