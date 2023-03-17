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
    matrix block assignment
    diag from matrix
    Identity
*/

namespace lager::gncpy::matrix {

template<typename T>
class Vector;

// row-major ordering
template<typename T>
class Matrix {
public:
    Matrix<T>() = default;

    Matrix<T>(const std::vector<size_t>& shape, const T* data=nullptr) {
        if(shape.size() < 1 || shape.size() > 2) {
            throw BadDimension("Only 2-D matrices are supported");
        }

        m_nRows = shape[0];
        m_nCols = shape[1];
        m_shape.emplace_back(m_nRows);
        m_shape.emplace_back(m_nCols);

        m_strides.emplace_back(m_shape[1]);
        m_strides.emplace_back(1);

        if(data != nullptr) {
            for(uint8_t ii = 0; ii < static_cast<uint8_t>(shape[0]*shape[1]); ii++) {
                m_data.emplace_back(data[ii]);
            }
        } else {
            for(uint8_t ii = 0; ii < static_cast<uint8_t>(shape[0]*shape[1]); ii++) {
                m_data.emplace_back(static_cast<T>(0));
            }
        }
    }

    /**
     * @brief Construct a new Matrix object
     * 
     * @param nRows 
     * @param nCols 
     * @param data 
     */
    Matrix<T>(uint8_t nRows, uint8_t nCols, std::vector<T> data) 
    : m_nRows(nRows),
    m_nCols(nCols),
    m_shape({m_nRows, m_nCols}),
    m_strides({m_nCols, 1}),
    m_data(data) {
        if(m_nRows * m_nCols != m_data.size()) {
            throw BadDimension("Supplied data does not match the given size");
        }
    }

    /**
     * @brief Construct a new Matrix object
     * 
     * @param listlist 
     */
    explicit Matrix<T>(std::initializer_list<std::initializer_list<T>> listlist)
    : m_nRows(listlist.size()),
    m_nCols(listlist.begin()->size()),
    m_shape({m_nRows, m_nCols}),
    m_strides({m_nCols, 1}) {
        if(m_nRows == 0 || m_nCols == 0) {
            throw BadDimension("Matrix constructor has size 0");
        }
    
        for (int r = 0; r < m_nRows; r++) {
            for (int c = 0; c < m_nCols; c++) {
                m_data.emplace_back(((listlist.begin()+r)->begin())[c]);
            }
        }
    }

    /**
     * @brief Construct a new empty Matrix object
     * 
     * @param nRows 
     * @param nCols 
     */
    Matrix<T>(uint8_t nRows, uint8_t nCols)
    : m_nRows(nRows),
    m_nCols(nCols),
    m_shape({m_nRows, m_nCols}),
    m_strides({m_nCols, 1}) {
        if(m_nRows == 0 || m_nCols == 0) {
            throw BadDimension("Matrix constructor has size 0");
        }

        for (int r = 0; r < m_nRows; r++) {
            for (int c = 0; c < m_nCols; c++) {
                m_data.emplace_back(static_cast<T>(0));
            }
        }

    }

    /**
     * @brief Matrix addition assignment
     * 
     * @param rhs 
     * @return Matrix& 
     */
    Matrix& operator+= (const Matrix& rhs) {
        if (!this->isSameSize(rhs)){
            throw BadDimension();
        }
        for (uint8_t i = 0; i < this->numRows() * this->numCols(); i++){
            m_data[i] += rhs.m_data[i];
        }

        return *this;
    }

    /**
     * @brief Matrix addition 
     * 
     * @param m 
     * @return Matrix 
     */
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

    /**
     * @brief 
     * 
     * @param rhs 
     * @return Matrix 
     */
    Matrix operator- (const Matrix& rhs) {
        if (!this->isSameSize(rhs)){
            throw BadDimension();
        }
        for (uint8_t i = 0; i < this->numRows() * this->numCols(); i++){
            m_data[i] -= rhs.m_data[i];
        }
    }

    /**
     * @brief Matrix multiplication
     * 
     * @param rhs 
     * @return Matrix 
     */
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

    /**
     * @brief Vector multiplication
     * 
     * @param rhs 
     * @return Vector<T> 
     */
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

    /**
     * @brief Matrix indexing
     * 
     * @param row 
     * @param col 
     * @return T& 
     */
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

    /**
     * @brief Matrix block indexing
     * 
     * @param start_row 
     * @param start_col 
     * @param row_span 
     * @param col_span 
     * @return Matrix 
     */
    Matrix operator() (uint8_t start_row, uint8_t start_col, uint8_t row_span, uint8_t col_span) const {
        if(start_row + row_span > m_nRows){
            throw BadIndex("Indexing outside rows");
        }
        if(start_col + col_span > m_nCols){
            throw BadIndex("Indexing outside columns");
        }
        std::vector<T> out;
        for(uint8_t r = start_row; r < start_row + row_span; r++) {
            for(uint8_t c = start_col; c < start_col + col_span; c++) {
                out.emplace_back(m_data[(this->rowColToLin(r,c))]);
            }
        }
        return Matrix(row_span, col_span, out);
    }

    template<typename R>
    friend std::ostream& operator<<(std::ostream& os, const Matrix<R>& m);

    inline std::vector<T>::iterator begin() noexcept { return m_data.begin(); }
    inline std::vector<T>::iterator end() noexcept { return m_data.end(); }
    inline std::vector<T>::const_iterator cbegin() const noexcept { return m_data.cbegin(); }
    inline std::vector<T>::const_iterator cend() const noexcept { return m_data.cend(); }

    /**
     * @brief Matrix transpose
     * 
     * @param in_place 
     * @return Matrix 
     */
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

    void LU_decomp(Matrix& L, Matrix& U){
        if (!this->isSameSize(L) || !this->isSameSize(U)){

        }
    }

    inline uint8_t numRows() const { return m_nRows; }
    inline uint8_t numCols() const { return m_nCols; }

    inline const T* data() const { return m_data.data(); }

    inline uint8_t size() const { return static_cast<uint8_t>(m_data.size()); }

    std::vector<uint8_t> shape() const { return m_shape; }
    std::vector<uint8_t> strides(bool bytes=false) const {
        std::vector<uint8_t> ret(m_strides.size());
        for(size_t ii = 0; ii < ret.size(); ii++) {
            ret[ii] = m_strides[ii];
            if(bytes) {
                ret[ii] *= sizeof(T);
            }
        }
        return ret;
    }

private:
    inline bool allowMultiplication(const Matrix& rhs) const {return numCols() == rhs.numRows();}
    inline bool isSameSize(const Matrix& rhs) const {return numRows() == rhs.numRows() && numCols() == rhs.numCols(); }
    inline uint8_t rowColToLin(uint8_t r, uint8_t c) const { return m_transposed ? r + m_nRows * c : c + m_nCols * r; }

    bool m_transposed = false;
    uint8_t m_nRows;
    uint8_t m_nCols;
    std::vector<uint8_t> m_shape;
    std::vector<uint8_t> m_strides;
    std::vector<T> m_data;
    
};

/**
 * @brief Print matrix using standard cout operator
 * 
 * @tparam T 
 * @param os 
 * @param m 
 * @return std::ostream& 
 */
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

template<typename T>
lager::gncpy::matrix::Matrix<T> identity(uint8_t n){
    lager::gncpy::matrix::Matrix<T> out( n, n);
    for (uint8_t i = 0; i < n; i++){
        out(i,i) = 1.;
    }
    return out;
}

} // namespace lager::gncpy::matrix
