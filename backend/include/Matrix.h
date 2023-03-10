#pragma once
#include <stdexcept>
#include <stdint.h>

namespace lager::gncpy::matrix {

class BadIndex final: public std::runtime_error {
public:
    explicit BadIndex(char const* const message) noexcept;
};


template<typename T>
class Matrix final{
public:
    explicit Matrix(std::initializer_list<std::initializer_list<T>> listlist)
    : m_nRows(listlist.begin()->size()),
    m_nCols(listlist.size()) {
        if(m_nRows == 0 || m_nCols == 0) {
            throw BadIndex("Matrix constructor has size 0");
        }
        m_data = new T[m_nRows * m_nCols];
    
        for (int r = 0; r < m_nRows; r++) {
            for (int c = 0; c < m_nCols; c++) {
                m_data[c + m_nCols * r] = ((listlist.begin()+r)->begin())[c];
            }
        }
    }

    Matrix(uint8_t nRows, uint8_t nCols)
    : m_nRows(nRows),
    m_nCols(nCols) {
        if(m_nRows == 0 || m_nCols == 0) {
            throw BadIndex("Matrix constructor has size 0");
        }
        m_data = new T[m_nRows * m_nCols];

        for (int r = 0; r < m_nRows; r++) {
            for (int c = 0; c < m_nCols; c++) {
                m_data[c + m_nCols * r] = static_cast<T>(0);
            }
        }

    }

    Matrix(const Matrix& m);               // Copy constructor

    ~Matrix() {
        delete[] m_data;
    }

    Matrix& operator= (const Matrix& m);   // Assignment operator
    Matrix& operator+= (const Matrix& rhs);

    Matrix operator+ (const Matrix& m);
    Matrix operator- (const Matrix& m);
    Matrix operator* (const Matrix& m);
    Matrix operator/ (const Matrix& m);

    T& operator() (uint8_t row, uint8_t col);
    T operator() (uint8_t row, uint8_t col) const;

    inline uint8_t numRows() const { return m_nRows; }
    inline uint8_t numCols() const { return m_nCols; }

private:
    inline bool allowMultiplication(const Matrix& rhs) const {}
    inline bool isSameSize(const Matrix& rhs) const {return numRows() == rhs.numRows && numCols() == rhs.numCols(); }

    uint8_t m_nRows;
    uint8_t m_nCols;
    T* m_data;
    
};

    
} // namespace lager::gncpy::matrix
