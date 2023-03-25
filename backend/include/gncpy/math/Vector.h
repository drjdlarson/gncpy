#pragma once
#include <algorithm>
#include "gncpy/math/Matrix.h"
#include "gncpy/math/Exceptions.h"


namespace lager::gncpy::matrix {

template<typename T>
class Vector final: public Matrix<T> {


public:
    Vector<T>()
    : Matrix<T>() {

    }

    Vector<T>(const std::vector<size_t>& shape, const T* data=nullptr)
    : Matrix<T>(shape, data) {

    }
    
    Vector<T>(size_t nElements, std::vector<T> data) 
    : Matrix<T>(nElements, 1, data) {
        
    }

    explicit Vector<T>(std::initializer_list<T> list) 
    : Matrix<T>(list.size(), 1, std::vector(list)) {

    }

    explicit Vector<T>(size_t nElements)
    : Matrix<T>(nElements, 1) {

    }

    T& operator() (size_t elem) {
        if(elem >= this->size()) {
            throw BadIndex("Indexing outside vector");
        }
        if(this->numRows() == 1){
            return Matrix<T>::operator()(0, elem);
        }
        return Matrix<T>::operator()(elem, 0);
    }

    T operator() (size_t elem) const {
        std::cout<<elem;
        if(elem >= this->size()) {
            throw BadIndex("Indexing outside vector");
        }
        if(this->numRows() == 1){
            return Matrix<T>::operator()(0, elem);
        }
        return Matrix<T>::operator()(elem, 0);
    }

    T magnitude() const {
        T sum = 0;
        for (auto const & i : *this){
                sum += i * i;
            }
        return T(sqrt(sum)); 
    }

    Vector<T> normalize(bool in_place = false) {
        T mag = this->magnitude();
        if (in_place){
            *this /= (mag);
            return *this;
        }
        else {
            std::vector<T> out;
            for (auto const & i : *this){
                out.emplace_back(i/mag);
            }
            return Vector<T> (out.size(), out);
        }
    }

    T dot(const Vector& rhs) const {
        if (!this->size() == rhs.size()){
            throw BadDimension("Vector size do not match");
        }
        T sum = 0;
        for (size_t i = 0; i < this->size(); i++){
            sum += this->operator()(i) * rhs(i);
        }
        return sum;
    }

    Vector<T> cross(const Vector& rhs) const {
        if (!this->size() == rhs.size()){
            throw BadDimension("Vector size do not match");
        }
        if (!this->size() > 3 || rhs.size() > 3){
            throw BadDimension("Can only do cross product on 3D vector");
        }
        std::vector<T> out {0,0,0};
        out[0] = static_cast <T> (this->operator()(1) * rhs(2) - this->operator()(2) * rhs(1));
        out[1] = static_cast <T> (this->operator()(2) * rhs(0) - this->operator()(0) * rhs(2));
        out[2] = static_cast <T> (this->operator()(0) * rhs(1) - this->operator()(1) * rhs(0));
        return Vector<T> (out.size(), out);
    }

    Matrix<T> skew() const{
        Matrix<T> out(3,3);
        out(0,1) = static_cast <T>(-this->operator()(2));
        out(0,2) = static_cast <T>(this->operator()(1));
        out(1,0) = static_cast <T>(this->operator()(2));
        out(1,2) = static_cast <T>(-this->operator()(0));
        out(2,0) = static_cast <T>(-this->operator()(1));
        out(2,1) = static_cast <T>(this->operator()(0));
        return out;
    }
    
};
    
} // namespace lager::gncpy::matrix
