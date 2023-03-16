#pragma once
#include <stdint.h>
#include <iostream>
#include <vector>
#include "gncpy/math/Exceptions.h"

/*
TODO
    Determinant
    Determinant unit test
    LU decomp 
    LU decomp unit test
    Inverse
        inplace and copy
    Inverse unit test
    Matrix * scalar
    scalar * matrix
    Matrix *= scalar
    matrix / scalar
    matrix /= scalar
    
*/

namespace lager::gncpy::matrix {

template<typename T>
class Vector;

// row-major ordering
template<typename T>
class Matrix {
public:
    Matrix(uint8_t nRows, uint8_t nCols, std::vector<T> data) 
    : m_nRows(nRows),
    m_nCols(nCols),
    m_data(data) {
        if(m_nRows * m_nCols != m_data.size()) {
            throw BadDimension("Supplied data does not match the given size");
        }
    }

    explicit Matrix(std::initializer_list<std::initializer_list<T>> listlist)
    : m_nRows(listlist.size()),
    m_nCols(listlist.begin()->size()) {
        if(m_nRows == 0 || m_nCols == 0) {
            throw BadDimension("Matrix constructor has size 0");
        }
    
        for (int r = 0; r < m_nRows; r++) {
            for (int c = 0; c < m_nCols; c++) {
                m_data.emplace_back(((listlist.begin()+r)->begin())[c]);
            }
        }
    }

    Matrix(uint8_t nRows, uint8_t nCols)
    : m_nRows(nRows),
    m_nCols(nCols) {
        if(m_nRows == 0 || m_nCols == 0) {
            throw BadDimension("Matrix constructor has size 0");
        }

        for (int r = 0; r < m_nRows; r++) {
            for (int c = 0; c < m_nCols; c++) {
                m_data.emplace_back(static_cast<T>(0));
            }
        }

    }

    Matrix& operator+= (const Matrix& rhs) {
        if (!this->isSameSize(rhs)){
            throw BadDimension();
        }
        for (uint8_t i = 0; i < this->numRows() * this->numCols(); i++){
            m_data[i] += rhs.m_data[i];
        }

        return *this;
    }

    Matrix operator+ (const Matrix& m) {
        if (!this->isSameSize(m)){
            throw BadDimension();
        }
        std::vector<T> out;
        for (uint8_t i = 0; i < this->numRows() * this->numCols(); i++){
            out.emplace_back(m_data[i] + m.m_data[i]);
        }

        return Matrix(m_nRows, m_nCols, out);
    }

    Matrix operator- (const Matrix& rhs) {
        if (!this->isSameSize(rhs)){
            throw BadDimension();
        }
        for (uint8_t i = 0; i < this->numRows() * this->numCols(); i++){
            m_data[i] -= rhs.m_data[i];
        }
    }

    Matrix operator* (const Matrix& rhs) {
        if(!this->allowMultiplication(rhs)) {
            throw BadDimension("Dimensions do not match");
        }

        std::vector<T> out;
        for(uint8_t r = 0; r < m_nRows; r++) {
            for(uint8_t c = 0; c < rhs.m_nCols; c++) {
                T total = 0;
                for(uint8_t k = 0; k < m_nCols; k++){
                    total += m_data[this->rowColToLin(r, k)] * rhs.m_data[this->rowColToLin(k, c)];
                }
                out.emplace_back(total);
            }
        }

        return Matrix(m_nRows, rhs.m_nCols, out);
    }

    Vector<T> operator* (const Vector<T>& rhs) {
        if(!this->allowMultiplication(rhs)) {
            throw BadDimension("Number of rows do not match");
        }

        std::vector<T> out;
        for(uint8_t r = 0; r < rhs.numRows(); r++) {
            T total = 0;
            for(uint8_t c = 0; c < m_nCols; c++) {
                total += m_data[this->rowColToLin(r, c)] * rhs.m_data[c];
            }
            out.emplace_back(total);
        }

        return Vector(out.size(), out);
    }

    Matrix operator/ (const Matrix& m);

    T& operator() (uint8_t row, uint8_t col) {
        if(row >= m_nRows) {
            throw BadIndex("Indexing outside rows.");
        }
        if(col >= m_nCols) {
            throw BadIndex("Indexing outside columns.");
        }
        return m_data[this->rowColToLin(row, col)];
    }

    T operator() (uint8_t row, uint8_t col) const {
        if(row >= m_nRows) {
            throw BadIndex("Indexing outside rows.");
        }
        if(col >= m_nCols) {
            throw BadIndex("Indexing outside columns.");
        }
        return m_data[this->rowColToLin(row, col)];
    }

    template<typename R>
    friend std::ostream& operator<<(std::ostream& os, const Matrix<R>& m);

    inline std::vector<T>::iterator begin() noexcept { return m_data.begin(); }
    inline std::vector<T>::iterator end() noexcept { return m_data.end(); }
    inline std::vector<T>::const_iterator cbegin() const noexcept { return m_data.cbegin(); }
    inline std::vector<T>::const_iterator cend() const noexcept { return m_data.cend(); }

    Matrix transpose(bool in_place = false) {
        if (in_place){
            uint8_t temp = this->m_nCols;
            this->m_nCols = this->m_nRows;
            this->m_nRows = temp;
            this->m_transposed = !this->m_transposed;
            return *this;
        }
        std::vector<T> out;
        for (uint8_t c = 0; c < this->numCols(); c++){
            for (uint8_t r = 0; r < this->numRows(); r++){
                out.emplace_back(this->m_data[this->rowColToLin(r,c)]);
            }
        }
        return Matrix(m_nCols, m_nRows, out);
    }

    inline uint8_t numRows() const { return m_nRows; }
    inline uint8_t numCols() const { return m_nCols; }
    inline bool beenTransposed() const { return m_transposed; }

    inline T* data() { return m_data.data(); }

    inline uint8_t size() const { return static_cast<uint8_t>(m_data.size()); }

    using row_t = uint8_t;
    using column_t = uint8_t;
    std::pair<row_t, column_t> shape() const { return std::make_pair(m_nRows, m_nCols); }

private:
    inline bool allowMultiplication(const Matrix& rhs) const {return numCols() == rhs.numRows();}
    inline bool isSameSize(const Matrix& rhs) const {return numRows() == rhs.numRows() && numCols() == rhs.numCols(); }
    inline uint8_t rowColToLin(uint8_t r, uint8_t c) const { return m_transposed ? r + m_nRows * c : c + m_nCols * r; }

    bool m_transposed = false;
    uint8_t m_nRows;
    uint8_t m_nCols;
    std::vector<T> m_data;
    
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& m){
    for (uint8_t r = 0; r < m.numRows(); r++){
                for (uint8_t c = 0; c < m.numCols(); c++){
            os << std::to_string(m(r,c))<< "\t";
        }
        os << "\n";
    }
    return os;
}

} // namespace lager::gncpy::matrix
