#pragma once
#include <stdint.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "gncpy/math/Exceptions.h"

/*
TODO
    Determinant unit test
    LU decomp unit test
    Inverse unit test
    matrix block assignment
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

    // mark as intentionally implicit to allow vector operator overloading
    explicit(false) operator Vector<T>() const {
        if(this->m_nRows == 1 || this->m_nCols == 1) {
            return Vector<T>(this->size(), this->m_data);
        } else {
            throw BadDimension("Matrix dimensions do not allow casting to vector");
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

    Matrix operator+ (const Matrix& m) const {
        if (!this->isSameSize(m)){
            throw BadDimension();
        }
        std::vector<T> out;
        for (uint8_t i = 0; i < this->numRows() * this->numCols(); i++){
            out.emplace_back(m_data[i] + m.m_data[i]);
        }

        return Matrix(m_nRows, m_nCols, out);
    }

    Matrix operator- (const Matrix& m) const {
        if (!this->isSameSize(m)){
            throw BadDimension();
        }
        std::vector<T> out;
        for (uint8_t i = 0; i < this->numRows() * this->numCols(); i++){
            out.emplace_back(m_data[i] - m.m_data[i]);
        }

        return Matrix(m_nRows, m_nCols, out);
    }

    Matrix& operator -= (const Matrix& rhs) {
        if(!this->isSameSize(rhs)) {
            throw BadDimension();
        }
        for(uint8_t r = 0; r < this->numRows(); r++) {
            for(uint8_t c = 0; c < this->numCols(); c++) {
                this->m_data[this->rowColToLin(r, c)] -= rhs(r, c);
            }
        }
        return *this;
    }

    Matrix operator* (const Matrix& rhs) const {
        if(!this->allowMultiplication(rhs)) {
            throw BadDimension("Dimensions do not match");
        }

        std::vector<T> out;
        for(uint8_t r = 0; r < m_nRows; r++) {
            for(uint8_t c = 0; c < rhs.m_nCols; c++) {
                T total = 0;
                for(uint8_t k = 0; k < m_nCols; k++){
                    total += m_data[this->rowColToLin(r, k)] * rhs.m_data[rhs.rowColToLin(k, c)];
                }
                out.emplace_back(total);
            }
        }

        return Matrix(m_nRows, rhs.m_nCols, out);
    }

    Matrix operator* (const T& scalar) const {
        std::vector<T> out = m_data;
        for (uint8_t i = 0; i < out.size(); i++){
            out[i] *= scalar;
        }
        return Matrix(m_nRows, m_nCols, out);
    }

    Matrix& operator*= (const T& scalar) {
        for (uint8_t i = 0; i < m_data.size(); i++){
            m_data[i] *= scalar;
        }
        return *this;
    }

    Vector<T> operator* (const Vector<T>& rhs) const {
        if(!this->allowMultiplication(rhs)) {
            throw BadDimension("Number of rows do not match");
        }

        std::vector<T> out;
        for(uint8_t r = 0; r < m_nRows; r++) {
            T total = 0;
            for(uint8_t c = 0; c < m_nCols; c++) {
                total += m_data[this->rowColToLin(r, c)] * rhs.m_data[c];
            }
            out.emplace_back(total);
        }

        return Vector(out.size(), out);
    }


    Matrix operator/ (const Matrix& m);

    Matrix operator/ (const T& scalar) const {
        std::vector<T> out = m_data;
        for (uint8_t i = 0; i < out.size(); i++){
            out[i] /= scalar;
        }
        return Matrix(m_nRows, m_nCols, out);
    }

    Matrix& operator/= (const T& scalar) {
        for (uint8_t i = 0; i < m_data.size(); i++){
            m_data[i] /= scalar;
        }
        return *this;
    }


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

    // Do as above for block assignment

    template<typename R>
    friend std::ostream& operator<<(std::ostream& os, const Matrix<R>& m);
    template<typename R>
    friend Matrix<R> operator* (const R& scalar, const Matrix<R>& m);

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
    Matrix<T> transpose(bool in_place) {
        if (in_place){
            uint8_t temp = this->m_nCols;
            this->m_nCols = this->m_nRows;
            this->m_nRows = temp;
            this->m_transposed = !this->m_transposed;
            return *this;
        }
        return this->transpose();
    }

    Matrix<T> transpose() const{
        std::vector<T> out;
        for (uint8_t c = 0; c < m_nCols; c++){
            for (uint8_t r = 0; r < m_nRows; r++){
                out.emplace_back(this->operator()(r,c));
            }
        }
        return Matrix<T>(m_nCols, m_nRows, out);
    }

    /**
     * @brief LU decomposition using Doolittle Algorithm
     * 
     * @param L 
     * @param U 
     */
    void LU_decomp(Matrix& L, Matrix& U) const{
        if (!this->isSameSize(L) || !this->isSameSize(U)||!this->isSquare()){
            throw BadDimension("Matrix dimension is invalid");
        }
        for (uint8_t i = 0; i < m_nCols; i++){
            // Constuct Upper matrix
            for (uint8_t j = i; j < m_nCols; j++){
                T sum = 0;
                for (uint8_t k = 0; k < i; k++){
                    sum += (L(i,k) * U (k,j));
                }
                U(i,j) = this->operator()(i,j) - sum;
            }

            // Construct Lower matrix 
            for (uint8_t j = i; j < m_nCols; j++){
                if (i==j){
                    L(i,i) = T(1);
                }
                else {
                    T sum = 0;
                    for (uint8_t k = 0; k < i; k++){
                        sum += (L(j,k) * U(k,i));
                    }
                    L(j,i) = (this->operator()(j,i) - sum) / U(i,i);
                }
            }
        }
    }

    /**
     * @brief Extract the diagonal elements of a square matrix
     * 
     * @return Vector<T> 
     */
    Vector<T> diag() const {
        if (!this->isSquare()){
            throw BadDimension ("Not a square matrix");
        }
        std::vector<T> out;
        for (uint8_t i = 0; i < m_nCols; i++){
            out.emplace_back(this->operator()(i,i));
        }
        return Vector(out.size(), out);
    }

    /**
     * @brief Calculate determinant of a matrix. Only square matrix implemented currently
     * 
     * @return T 
     */
    T determinant() const{
        if (!this->isSquare()){
            throw BadDimension("Non square matrix non implimented");
        }
        Matrix L(m_nCols,m_nCols);
        Matrix U(m_nCols,m_nCols);
        this->LU_decomp(L,U);
        return this->determinant(U);
    }

    T determinant(Matrix& U) const{
        if (!this->isSquare()){
            throw BadDimension("Non square matrix non implimented");
        }
        Vector<T> u_diag = U.diag();
        T det = u_diag(0);
        for (uint8_t i = 1; i < u_diag.size(); i++){
            det *= u_diag(i);
        }
        return det;
    }

    /**
     * @brief Calculate the inverse of a matrix using LU linear solver. Only square matrix implemented currently
     * 
     * @return Matrix 
     */
    Matrix inverse() const{
        if (!this->isSquare()){
            throw BadDimension("Non square matrix non implimented");
        }
        Matrix L(m_nCols,m_nCols);
        Matrix U = L;
        this->LU_decomp(L,U);
        T det = this->determinant(U);
        if (det == static_cast<T>(0)){
            throw BadDimension("Matrix is singular");
        }
        return LU_solve(L, U, identity<T>(m_nCols));
    }

    inline T toScalar() const {
        if(this->size() != 1) {
            throw BadDimension("Matrix has too many values to convert to scalar");
        }
        return this->m_data[0];
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
    inline bool isSquare () const {return m_nCols == m_nRows;}
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

template<typename T>
    lager::gncpy::matrix::Matrix<T> operator* (const T& scalar, const lager::gncpy::matrix::Matrix<T>& m){
        lager::gncpy::matrix::Matrix<T> out = m * scalar;
        return out;
    }


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

/**
 * @brief Construct identity matrix
 * 
 * @tparam T 
 * @param n 
 * @return lager::gncpy::matrix::Matrix<T> 
 */
template<typename T>
lager::gncpy::matrix::Matrix<T> identity(uint8_t n){
    lager::gncpy::matrix::Matrix<T> out( n, n);
    for (uint8_t i = 0; i < n; i++){
        out(i,i) = T(1);
    }
    return out;
}

/**
 * @brief Perform forward substitution to solve linear system of equation given unity lower triangle matrix 
 * 
 * @tparam T 
 * @param L 
 * @param b 
 * @return lager::gncpy::matrix::Matrix<T> 
 */
template<typename T>
lager::gncpy::matrix::Matrix<T> forward_sub(const lager::gncpy::matrix::Matrix<T>& L, 
    const lager::gncpy::matrix::Matrix<T>& b){
    if (!L.numCols() == b.numRows()){
        throw BadDimension("Invalid matrix dimension");
    }
    lager::gncpy::matrix::Matrix<T> x (b.numRows(),b.numCols());
    for (uint8_t k = 0; k < b.numCols(); k++){
        for (uint8_t i = 0; i < b.numRows(); i++){
            T sum = b(i,k);
            for (uint8_t j = 0; j < i; j++){
                sum -= (L(i,j) * x(j,k));
            }
            /*
            Division is skipped here assuming that the Doolittle is called 
            which yield a unitary Lower matrix. Should save some time on division
            */
            //x(i,k) = sum / L(i,i);
            x(i,k) = sum;
        }
    }
    return x;
}

/**
 * @brief Perform back substitution to solve linear system of equation given upper triangle matrix
 * 
 * @tparam T 
 * @param U 
 * @param b 
 * @return lager::gncpy::matrix::Matrix<T> 
 */

template<typename T>
lager::gncpy::matrix::Matrix<T> back_sub(const lager::gncpy::matrix::Matrix<T>& U, 
    const lager::gncpy::matrix::Matrix<T>& b){
    if (!U.numCols() == b.numRows()){
        throw BadDimension("Invalid matrix dimension");
    }
    lager::gncpy::matrix::Matrix<T> x (b.numRows(),b.numCols());
    for (uint8_t k = 0; k < b.numCols(); k++){
        for (int8_t i = b.numRows()-1; i > -1; i--){
            T sum = b(i,k);
            for (int8_t j = b.numRows()-1; j > i; j--){
                sum -= (U(i,j) * x(j,k));
            }
            x(i,k) = sum / U(i,i);
        }
    }
    return x;
}
    
/**
 * @brief Sovlve system of linear equation from LU matrices
 * 
 * @tparam T 
 * @param L 
 * @param U 
 * @param b 
 * @return lager::gncpy::matrix::Matrix<T> 
 */
template<typename T>
lager::gncpy::matrix::Matrix<T> LU_solve(const lager::gncpy::matrix::Matrix<T>& L, 
    const lager::gncpy::matrix::Matrix<T> U, const lager::gncpy::matrix::Matrix<T>& b){
    if (!L.numCols() == b.numRows() || !U.numCols() == b.numRows()){
        throw BadDimension("Invalid vector dimension");
    }
    lager::gncpy::matrix::Matrix<T> out = forward_sub(L,b);
    out = back_sub(U,out);
    return out;
}

} // namespace lager::gncpy::matrix
