#pragma once
#include <stdint.h>
#include "gncpy/math/Exceptions.h"

namespace lager::gncpy::matrix {


// row-major ordering
template<typename T>
class Matrix final{
public:
    explicit Matrix(std::initializer_list<std::initializer_list<T>> listlist)
    : m_nRows(listlist.begin()->size()),
    m_nCols(listlist.size()) {
        if(m_nRows == 0 || m_nCols == 0) {
            throw BadIndex("Matrix constructor has size 0");
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
            throw BadIndex("Matrix constructor has size 0");
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
        for (uint8_t i = 0; i < this->numRows() * this->numCols(); i++){
            m_data[i] += m.m_data[i];
        }
    }

    Matrix operator- (const Matrix& m) {
        if (!this->isSameSize(m)){
            throw BadDimension();
        }
        for (uint8_t i = 0; i < this->numRows() * this->numCols(); i++){
            m_data[i] -= m.m_data[i];
        }
    }

    Matrix operator* (const Matrix& m);
    Matrix operator/ (const Matrix& m);

    T& operator() (uint8_t row, uint8_t col);
    T operator() (uint8_t row, uint8_t col) const;

    inline uint8_t numRows() const { return m_nRows; }
    inline uint8_t numCols() const { return m_nCols; }

private:
    inline bool allowMultiplication(const Matrix& rhs) const {return numCols() == rhs.numRows();}
    inline bool isSameSize(const Matrix& rhs) const {return numRows() == rhs.numRows() && numCols() == rhs.numCols(); }

    uint8_t m_nRows;
    uint8_t m_nCols;
    std::vector<T> m_data;
    
};

} // namespace lager::gncpy::matrix
