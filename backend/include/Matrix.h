#pragma once
#include <stdexcept>
#include <stdint.h>

namespace lager::gncpy::matrix {

class BadIndex final: public std::runtime_error {
public:
    BadIndex(char const* const message) throw();
};

class BadDimension final: public std::runtime_error {
public:
    BadDimension(char const* const message) throw();
};

template<typename T>
class Matrix final{
public:
    Matrix(uint8_t nRows, uint8_t nCols)
    : m_nRows(nRows),
    m_nCols(nCols) {
        if(m_nRows == 0 || m_nCols == 0) {
            throw BadIndex("Matrix constructor has size 0");
        }
        m_data = new T[m_nRows * m_nCols];

    }

    Matrix(const Matrix& m);               // Copy constructor

    ~Matrix() {
        delete[] m_data;
    }

    Matrix& operator= (const Matrix& m);   // Assignment operator

    Matrix operator+ (const Matrix& m);
    Matrix operator- (const Matrix& m);
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
    T* m_data;
    
};

    
} // namespace lager::gncpy::matrix
