#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <math.h>
#include <mkl.h>

#ifndef __MATRIX__
#define __MATRIX__
namespace py = pybind11;

class BlockMy {
public:
    BlockMy(size_t nrow, size_t ncol, bool colmajor);
    BlockMy(const BlockMy &block);
    ~BlockMy();
    double   operator() (size_t row, size_t col) const;
    void setContent(double *ptr, size_t row_stride) ;

    size_t nrow() const {return m_nrow;}
    size_t ncol() const {return m_ncol;}

private:
    size_t m_nrow;
    size_t m_ncol;
    double *m_buffer;
    size_t m_row_stride;
    bool m_colmajor;

};

class Matrix {
public:
    Matrix();
    Matrix(size_t nrow, size_t ncol);
    static Matrix fillwith(size_t nrow, size_t ncol, double num);
    
    template<typename Type>
    Matrix(Type* ptr, size_t nrow, size_t ncol);
    Matrix(const Matrix &target);
    ~Matrix();
    
    ///////////////////////////////////////////
    // Math Operation Function Start
    ///////////////////////////////////////////
    double   operator() (size_t row, size_t col) const;
    double & operator() (size_t row, size_t col) ;

    Matrix  operator+(const Matrix &mat) const ;
    Matrix& operator+=(const Matrix &mat) ;
    Matrix  operator+(double num) const;
    Matrix& operator+=(double num);
    friend Matrix operator+(double num, const Matrix &mat);

    Matrix  operator-(const Matrix &mat) const ;
    Matrix& operator-=(const Matrix &mat) ;
    Matrix  operator-(double num) const;
    Matrix& operator-=(double num);
    friend Matrix operator-(double num, const Matrix &mat);

    Matrix  operator*(const Matrix &mat) const;
    Matrix& operator*=(const Matrix &mat);
    Matrix  operator*(double num) const;
    Matrix& operator*=(double num);
    friend Matrix operator*(double num, const Matrix &mat);

    Matrix  operator/(const Matrix &mat) const;
    Matrix& operator/=(const Matrix &mat) ;
    Matrix  operator/(double num) const;
    Matrix& operator/=(double num);
    friend Matrix operator/(double num, const Matrix &mat);
    
    bool operator==(const Matrix &target) const;
    void operator=(const Matrix &target) ;

    Matrix power(double p) const;
    Matrix exp() const;
    Matrix log() const;
    Matrix sigmoid() const;
    Matrix relu() const;
    ///////////////////////////////////////////
    // Math Operation Function End
    ///////////////////////////////////////////

    BlockMy get_block(size_t block_size, size_t row_idx, size_t col_idx, bool col2row = false) const;
    void set_block(size_t block_size, size_t row_idx, size_t col_idx, const Matrix &mat) ;

    Matrix T() const;
    double *data() { return m_buffer; }
    size_t nrow() const { return m_nrow; }
    size_t ncol() const { return m_ncol; }
    double* get_buffer() const {return m_buffer;}
    py::array_t<double, py::array::c_style | py::array::forcecast> get_array();

    friend Matrix multiply_mkl(const Matrix &mat1, const Matrix &mat2);
    void print_shape(const char* mat_name="") const {std::cout << mat_name << " m_nrow: " << m_nrow << " m_ncol: " << m_ncol << std::endl; }
public:
    size_t m_nrow;
    size_t m_ncol;
    double * m_buffer;
public:
    static int multiplication_mode;

};

///////////////////////////////////////////////////////////
// Set and Get Matrix Multiplication Core
// 1 Naive Method
// 2 MKL Library
// 3 Tile Method
///////////////////////////////////////////////////////////
void SetMatrixMode(int val);
int GetMatrixMode();
void test(py::buffer b);

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
Matrix mat_multiply(const Matrix &mat1, const Matrix &mat2);
Matrix multiply_naive(const Matrix &mat1, const Matrix &mat2);
Matrix multiply_mkl(const Matrix &mat1, const Matrix &mat2);
Matrix multiply_tile_modify(const Matrix &mat1, const Matrix &mat2, size_t block_size);

// ******************************************************
// Not Correct if the row and col are 2's multipliers
// ******************************************************
Matrix multiply_tile(const Matrix &mat1, const Matrix &mat2, size_t block_size);
Matrix multiply_tile_nb(const Matrix &mat1, const Matrix &mat2, size_t block_size);
Matrix multiply_tile_nb_reorder(const Matrix &mat1, const Matrix &mat2, size_t block_size);
#endif
// end #define __MATRIX__