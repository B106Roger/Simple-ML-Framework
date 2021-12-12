#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <math.h>
#include <mkl.h>

namespace py = pybind11;

class Block {
public:
    Block(size_t nrow, size_t ncol, bool colmajor);
    Block(const Block &block);
    ~Block();
    double   operator() (size_t row, size_t col) const;
    void setContent(double *ptr, size_t row_stride) ;

    size_t nrow() const {return m_nrow;}
    size_t ncol() const {return m_ncol;}

private:
    bool m_colmajor;
    double *m_buffer;
    size_t m_row_stride;
    size_t m_nrow;
    size_t m_ncol;
};

class Matrix {
public:
    Matrix():
        m_buffer(NULL), m_nrow(0), m_ncol(0)
    {
        
    }
    Matrix(size_t nrow, size_t ncol);

    template<typename T>
    Matrix(T* ptr, size_t nrow, size_t ncol);

    Matrix(Matrix const &target) ;

    ~Matrix();

    // No bound check.
    double   operator() (size_t row, size_t col) const;
    double & operator() (size_t row, size_t col) ;
    Matrix operator+(const Matrix &mat) const ;
    void operator+=(const Matrix &mat) ;
    Matrix operator-(const Matrix &mat) const ;
    void operator-=(const Matrix &mat) ;
    void operator=(const Matrix &target) ;
    bool operator==(const Matrix &target) const;

    Block get_block(size_t block_size, size_t row_idx, size_t col_idx, bool col2row = false) const;

    void set_block(size_t block_size, size_t row_idx, size_t col_idx, const Matrix &mat) ;

    double *data() { return m_buffer; }
    size_t nrow() const { return m_nrow; }
    size_t ncol() const { return m_ncol; }
    friend Matrix multiply_mkl(Matrix &mat1, Matrix &mat2);

private:
    
    size_t m_nrow;
    size_t m_ncol;
    double * m_buffer;

};
