#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <math.h>
#include <mkl.h>
#include <pthread.h>
#include <ctime>
#include <cstring>
#include <mkl.h>
#include <omp.h>
#include "matrix.h"
#include "debug.h"
#include "tiler.h"

// Matrix = double + Matrix
#define OPERATOR_DOUBLE_MATRIX(FUNCNAME,OPT) \
Matrix FUNCNAME(double num, const Matrix &mat) \
{\
    Matrix result(mat);\
    for(size_t i = 0; i < result.m_nrow*result.m_ncol; i++){\
        result.m_buffer[i] = num OPT result.m_buffer[i];\
    }\
    return result;\
}\

// Matrix = Matrix + double
#define OPERATOR_MATRIX_DOUBLE(FUNCNAME,OPT)\
Matrix Matrix::FUNCNAME(double num) const\
{\
    Matrix result(m_nrow,m_ncol);\
    for(size_t i = 0; i < m_nrow*m_ncol; i++){\
        result.m_buffer[i] = m_buffer[i] OPT num;\
    }\
    return result;\
}\

// Matrix = Matrix + Matrix
#define OPERATOR_MATRIX_MATRIX(FUNCNAME,OPT)\
Matrix Matrix::FUNCNAME(const Matrix &mat) const \
{\
    size_t row=std::max(mat.m_nrow, m_nrow);\
    size_t col=std::max(mat.m_ncol, m_ncol);\
    if (DEBUG) {\
    if (mat.m_nrow != m_nrow && !(mat.m_nrow == 1 || m_nrow == 1)) \
        throw std::runtime_error("The shape is not broadcastable in row.");\
    else if (mat.m_ncol != m_ncol && !(mat.m_ncol == 1 || m_ncol == 1))\
        throw std::runtime_error("The shape is not broadcastable in column.");\
    }\
    Matrix result(row, col);\
    if (mat.m_nrow == m_nrow && mat.m_ncol == m_ncol) {\
        for(size_t i = 0; i < m_nrow*m_ncol; i++){\
            result.m_buffer[i] = m_buffer[i] OPT mat.m_buffer[i];\
        }\
        return result;\
    } else {\
        for (size_t i = 0; i < row; i+=1) {\
            for (size_t j = 0; j < col; j+=1) {\
                result.m_buffer[i*col+j] = \
                    (*this)(i % m_nrow,     j % m_ncol) OPT \
                        mat(i % mat.m_nrow, j % mat.m_ncol);\
            }\
        }\
        return result;\
    }\
}\

// Matrix += double
#define OPERATOR_ASSIGN_MATRIX_DOUBLE(FUNCNAME,OPT)\
Matrix& Matrix::FUNCNAME(double num) \
{\
    for(size_t i = 0; i < m_nrow*m_ncol; i++){\
        m_buffer[i] OPT num;\
    }\
    return *this;\
}\

// Matrix += Matrix
#define OPERATOR_ASSIGN_MATRIX_MATRIX(FUNCNAME,OPT)\
Matrix& Matrix::FUNCNAME(const Matrix &mat) {\
    for (size_t i=0; i< m_nrow; i+=1) {\
        for (size_t j=0; j<m_ncol; j+=1) {\
            m_buffer[i*m_ncol+j] OPT mat(i,j);\
        }\
    }\
    return *this;\
}\

namespace py = pybind11;


////////////////////////////////////////////////////////////////////////////////
/////////////       Class Member for Matrix Class                     //////////
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////
// Constructor Start
////////////////////////////////////////
int Matrix::multiplication_mode = 2;

Matrix::Matrix()
    : m_nrow(0), m_ncol(0), m_buffer(NULL)
{
}

Matrix::Matrix(size_t nrow, size_t ncol)
  : m_nrow(nrow), m_ncol(ncol), m_buffer(NULL)
{
    size_t nelement = nrow * ncol;
    // std::cout << "m_nrow: " << m_nrow << " m_ncol: " << m_ncol <<std::endl;
    m_buffer = new double[nelement];
    // std::cout << "nelement: " << nelement << " ptr: " << m_buffer << std::endl;
    if (m_buffer!=NULL)
        memset(m_buffer, 0, nelement*sizeof(double));
    else 
        std::cout << "Your buffer is out of memory" << std::endl;
}

template<typename Type>
Matrix::Matrix(Type* ptr, size_t nrow, size_t ncol)
    :m_nrow(nrow), m_ncol(ncol), m_buffer(NULL)
{
    
    size_t nelement = nrow * ncol;
    m_buffer = new double[nelement];
    for(size_t i =0; i < nelement; i++) 
    {
        m_buffer[i] = (double)ptr[i];
    }
}

Matrix::Matrix(const Matrix &target) 
{
    m_nrow=target.nrow();
    m_ncol=target.ncol();
    m_buffer = new double[m_nrow*m_ncol];
    memcpy(m_buffer, target.m_buffer, sizeof(double) * m_nrow * m_ncol);
}

Matrix Matrix::fillwith(size_t nrow, size_t ncol, double num) 
{
    Matrix mat(nrow, ncol);
    std::fill(mat.m_buffer, mat.m_buffer+nrow*ncol, num);
    return mat;
}

Matrix::~Matrix() 
{ 
    if (m_buffer == NULL && m_nrow != 0 and m_ncol != 0)
        std::cout << "pointer is null but has m_row, m_col is not 0\n" << std::endl;

    if (m_buffer != NULL)
    {
        delete[] m_buffer; 
        m_nrow=m_ncol = 0;
    }
}

///////////////////////////////////////
// Other Function
///////////////////////////////////////
// No bound check.
double   Matrix::operator() (size_t row, size_t col) const // for setitem
{
    if (row > m_nrow)
        throw std::runtime_error("row out of bound");
    if (col > m_ncol)
        throw std::runtime_error("col out of bound");
    return m_buffer[row*m_ncol + col];
}
double & Matrix::operator() (size_t row, size_t col) // for setitem
{
    if (row > m_nrow)
        throw std::runtime_error("row out of bound");
    if (col > m_ncol)
        throw std::runtime_error("col out of bound");
    return m_buffer[row*m_ncol + col];
}
// implement in vectorize mode

OPERATOR_DOUBLE_MATRIX(operator+,+)
OPERATOR_DOUBLE_MATRIX(operator-,-)
OPERATOR_DOUBLE_MATRIX(operator*,*)
OPERATOR_DOUBLE_MATRIX(operator/,/)

OPERATOR_MATRIX_DOUBLE(operator+,+)
OPERATOR_MATRIX_DOUBLE(operator-,-)
OPERATOR_MATRIX_DOUBLE(operator*,*)
OPERATOR_MATRIX_DOUBLE(operator/,/)

OPERATOR_MATRIX_MATRIX(operator+,+)
OPERATOR_MATRIX_MATRIX(operator-,-)
OPERATOR_MATRIX_MATRIX(operator*,*)
OPERATOR_MATRIX_MATRIX(operator/,/)

OPERATOR_ASSIGN_MATRIX_DOUBLE(operator+=,+=)
OPERATOR_ASSIGN_MATRIX_DOUBLE(operator-=,-=)
OPERATOR_ASSIGN_MATRIX_DOUBLE(operator*=,*=)
OPERATOR_ASSIGN_MATRIX_DOUBLE(operator/=,/=)

OPERATOR_ASSIGN_MATRIX_MATRIX(operator+=,+=)
OPERATOR_ASSIGN_MATRIX_MATRIX(operator-=,-=)
OPERATOR_ASSIGN_MATRIX_MATRIX(operator*=,*=)
OPERATOR_ASSIGN_MATRIX_MATRIX(operator/=,/=)

void Matrix::operator=(const Matrix &target) 
{
    if (m_buffer != NULL)
        delete[] m_buffer;
    m_nrow=target.nrow();
    m_ncol=target.ncol();
    m_buffer = new double[m_nrow*m_ncol];
    // std::cout << "operator= ptr: " << m_buffer << std::endl;
    memcpy(m_buffer, target.m_buffer, sizeof(double) * m_nrow * m_ncol);
}
bool Matrix::operator==(const Matrix &target) const
{
    if (m_nrow != target.m_nrow || m_ncol != target.m_ncol) {
        return false;
    } else {
        for (size_t i = 0; i < m_nrow; i++) {
            for (size_t j = 0; j < m_ncol; j++) {
                if ((*this)(i,j) != target(i,j)) return false;
            }
        }
        return true;
    }
}

Matrix Matrix::power(double p) const 
{
    size_t nelement=m_ncol*m_nrow;
    Matrix result(m_nrow, m_ncol);
    for(size_t i = 0; i < nelement; i++){
        result.m_buffer[i] = std::pow(m_buffer[i], p);
    }
    return result;
}
Matrix Matrix::exp() const 
{
    size_t nelement=m_ncol*m_nrow;
    Matrix result(m_nrow, m_ncol);
    for(size_t i = 0; i < nelement; i++){
        result.m_buffer[i] = std::exp(m_buffer[i]);
    }
    return result;
}
Matrix Matrix::log() const 
{
    size_t nelement=m_ncol*m_nrow;
    Matrix result(m_nrow, m_ncol);
    for(size_t i = 0; i < nelement; i++){
        result.m_buffer[i] = std::log(m_buffer[i]);
    }
    return result;
}
Matrix Matrix::sigmoid() const 
{
    Matrix result(*this);
    for (size_t i = 0; i < m_nrow*m_ncol; i++) {
        result.m_buffer[i] = 1.0 / (1.0 + std::exp(-result.m_buffer[i]));
    }
    return result;
}
Matrix Matrix::relu() const 
{
    Matrix result(*this);
    for(size_t i = 0; i < result.m_nrow * result.m_ncol; i++) {
        if (result.m_buffer[i] < 0.l)
            result.m_buffer[i] = 0.l;
    }
    return result;
}
Matrix Matrix::T() const 
{
    Matrix result(m_ncol, m_nrow);
    for(size_t i = 0; i<m_nrow*m_ncol; i++)
    {
        size_t col_idx = i / m_ncol;
        size_t row_idx = i % m_ncol;
        result(row_idx, col_idx) = m_buffer[i];
    }
    // std::cout << "right before T return: " << result.get_buffer() << std::endl;
    return result;
}

py::array_t<double, py::array::c_style | py::array::forcecast> Matrix::get_array()
{
    py::buffer_info  buffer(
        m_buffer,
        sizeof(double),
        py::format_descriptor<double>::format(),
        2,
        {m_nrow, m_ncol},
        {sizeof(double) * m_ncol, sizeof(double)}
    );
    return py::array_t<double, py::array::c_style | py::array::forcecast>(buffer, py::cast(this));
}


//////////////////////////////////////////////////////////////////////
// Matrix Multiplication Core
//////////////////////////////////////////////////////////////////////

// Tested Part
Matrix mat_multiply(const Matrix &mat1, const Matrix &mat2)
{
    switch (Matrix::multiplication_mode)
    {
        case 1:
            return multiply_naive(mat1, mat2);
        case 2:
            return multiply_mkl(mat1, mat2);
        case 3: 
            return multiply_tile_modify(mat1, mat2, 16);
        case 4:
            return multiply_tile_modify_pthread(mat1, mat2, 16, 8, 8);
        case 5:
            return multiply_tile_SIMD_SSE(mat1, mat2, 16);        
        case 6:
            return multiply_tile_SIMD_AVX(mat1, mat2, 16);
    }
    return multiply_naive(mat1, mat2);
}

Matrix multiply_naive(const Matrix &mat1, const Matrix &mat2) 
{
    size_t row=mat1.nrow();
    size_t col=mat2.ncol();
    size_t content=mat1.ncol();
    Matrix tmp(row, col);
    for (size_t i=0; i<row; i++) {
        for (size_t j=0; j<col; j++) {
            double sum=0.0;
            for (size_t k=0; k<content; k++) {
                sum+=mat1(i,k)*mat2(k,j);
            }
            tmp(i,j)=sum;
        }
    }
    return tmp;
}

Matrix multiply_naive_reorder(const Matrix &mat1, const Matrix &mat2) 
{
    size_t row=mat1.nrow();
    size_t col=mat2.ncol();
    size_t content=mat1.ncol();
    Matrix tmp(row, col);
    for (size_t i=0; i<row; i++) {
        for (size_t k=0; k<content; k++) {
            for (size_t j=0; j<col; j++) {
                tmp(i,j)+=mat1(i,k)*mat2(k,j);
            }
        }
    }
    return tmp;
}

Matrix multiply_mkl(const Matrix &mat1, const Matrix &mat2) 
{
    if (mat1.ncol() != mat2.nrow())
        throw std::runtime_error("mismatch mat1.ncol and mat2.nrow!");
    mkl_set_num_threads(1);
    Matrix ret(mat1.nrow(), mat2.ncol());
    cblas_dgemm(
        CblasRowMajor /* const CBLAS_LAYOUT Layout */
      , CblasNoTrans /* const CBLAS_TRANSPOSE transa */
      , CblasNoTrans /* const CBLAS_TRANSPOSE transb */
      , mat1.nrow() /* const MKL_INT m */
      , mat2.ncol() /* const MKL_INT n */
      , mat1.ncol() /* const MKL_INT k */
      , 1.0 /* const double alpha */
      , mat1.m_buffer /* const double *a */
      , mat1.ncol() /* const MKL_INT lda */
      , mat2.m_buffer /* const double *b */
      , mat2.ncol() /* const MKL_INT ldb */
      , 0.0 /* const double beta */
      , ret.m_buffer /* double * c */
      , ret.ncol() /* const MKL_INT ldc */
    );
    return ret;
}

Matrix multiply_tile_modify(const Matrix &mat1, const Matrix &mat2, size_t block_size) 
{
    Matrix ret(mat1.nrow(), mat2.ncol());

    const size_t tsize = block_size;

    const size_t nrow1 = mat1.nrow();
    const size_t ncol1 = mat1.ncol();
    // const size_t nrow2 = mat2.nrow();
    const size_t ncol2 = mat2.ncol();

    const size_t ntrow1 = nrow1 / tsize;
    const size_t ntcol1 = ncol1 / tsize;
    // const size_t ntrow2 = nrow2 / tsize;
    const size_t ntcol2 = ncol2 / tsize;

    const size_t row_spare_size = nrow1 % tsize;
    const size_t con_spare_size = ncol1 % tsize;
    const size_t col_spare_size = ncol2 % tsize;

    const size_t row_flag = row_spare_size > 0;
    const size_t con_flag = con_spare_size > 0;
    const size_t col_flag = col_spare_size > 0;

    Block value(tsize);
    Tiler tiler(tsize);
    // clock_t s2,e2;
    // double avg_load = 0.0;
    // int cnt = 0;
    // std::cout << row_flag << " " << con_flag << " " << col_flag << std::endl;
    for (size_t it=0; it<ntrow1+row_flag; ++it)
    {
        size_t tile_row_size = (row_flag & (ntrow1 == it)) ? row_spare_size : tsize; 
        for (size_t kt=0; kt<ntcol2+col_flag; ++kt)
        {
            size_t tile_col_size = (col_flag & (ntcol2 == kt)) ? col_spare_size : tsize; 
            // s2 = clock();
            value = 0;
            for (size_t jt=0; jt<ntcol1+con_flag; ++jt)
            {
                size_t tile_con_size = (con_flag & (ntcol1 == jt)) ? con_spare_size : tsize;
                tiler.load(
                    mat1, it, jt, tile_row_size, tile_con_size, 
                    mat2, jt, kt, tile_con_size, tile_col_size
                );
                tiler.multiply(tile_row_size, tile_col_size, tile_con_size);
                value += (*tiler.m_ret);

            }

            value.save(ret, it, kt, tile_row_size, tile_col_size);
            // e2 = clock();
            // avg_load += (double)(e2-s2) / CLOCKS_PER_SEC;
            // cnt++;
        }
    }
    // printf("avg work load = %f cnt = %d\n", avg_load/cnt, cnt);
    return ret;
}

// Accelerate Part
//////////////////////////////////////////////////////////////////////////////////////////
// Create Your Own Matrix Multiplication Below
// Note that all the Matrix Multiplication should have signature like
// Matrix multiply_YOUR_FUNC_NAME(const Matrix &mat1, const Matrix &mat2, ...other-argument) 
//////////////////////////////////////////////////////////////////////////////////////////

Matrix multiply_tile_SIMD_SSE(const Matrix &mat1, const Matrix &mat2, size_t block_size) 
{
    Matrix ret(mat1.nrow(), mat2.ncol());

    const size_t tsize = block_size;

    const size_t nrow1 = mat1.nrow();
    const size_t ncol1 = mat1.ncol();
    // const size_t nrow2 = mat2.nrow();
    const size_t ncol2 = mat2.ncol();

    const size_t ntrow1 = nrow1 / tsize;
    const size_t ntcol1 = ncol1 / tsize;
    // const size_t ntrow2 = nrow2 / tsize;
    const size_t ntcol2 = ncol2 / tsize;

    const size_t row_spare_size = nrow1 % tsize;
    const size_t con_spare_size = ncol1 % tsize;
    const size_t col_spare_size = ncol2 % tsize;

    const size_t row_flag = row_spare_size > 0;
    const size_t con_flag = con_spare_size > 0;
    const size_t col_flag = col_spare_size > 0;

    Block value(tsize);
    Tiler tiler(tsize);
    // std::cout << row_flag << " " << con_flag << " " << col_flag << std::endl;
    for (size_t it=0; it<ntrow1+row_flag; ++it)
    {
        size_t tile_row_size = (row_flag & (ntrow1 == it)) ? row_spare_size : tsize; 
        for (size_t kt=0; kt<ntcol2+col_flag; ++kt)
        {
            size_t tile_col_size = (col_flag & (ntcol2 == kt)) ? col_spare_size : tsize; 
            value = 0;
            for (size_t jt=0; jt<ntcol1+con_flag; ++jt)
            {
                size_t tile_con_size = (con_flag & (ntcol1 == jt)) ? con_spare_size : tsize; 
                // std::cout 
                // << " tile_row_size(" << it << "," << ntrow1 << "," << tile_row_size << ") "
                // << " tile_col_size(" << kt << "," << ntcol2 << "," << tile_col_size << ") "
                // << " tile_con_size(" << jt << "," << ntcol1 << "," << tile_con_size << ") "
                // << " " << row_flag << " " << con_flag << " " << col_flag << std::endl;
                tiler.load(
                    mat1, it, jt, tile_row_size, tile_con_size, 
                    mat2, jt, kt, tile_con_size, tile_col_size
                );
                tiler.SSE_multiply(tile_row_size, tile_col_size, tile_con_size);
                value += (*tiler.m_ret);

            }

            value.save(ret, it, kt, tile_row_size, tile_col_size);
        }
    }
    return ret;
}

Matrix multiply_tile_SIMD_AVX(const Matrix &mat1, const Matrix &mat2, size_t block_size) 
{
    Matrix ret(mat1.nrow(), mat2.ncol());

    const size_t tsize = block_size;

    const size_t nrow1 = mat1.nrow();
    const size_t ncol1 = mat1.ncol();
    // const size_t nrow2 = mat2.nrow();
    const size_t ncol2 = mat2.ncol();

    const size_t ntrow1 = nrow1 / tsize;
    const size_t ntcol1 = ncol1 / tsize;
    // const size_t ntrow2 = nrow2 / tsize;
    const size_t ntcol2 = ncol2 / tsize;

    const size_t row_spare_size = nrow1 % tsize;
    const size_t con_spare_size = ncol1 % tsize;
    const size_t col_spare_size = ncol2 % tsize;

    const size_t row_flag = row_spare_size > 0;
    const size_t con_flag = con_spare_size > 0;
    const size_t col_flag = col_spare_size > 0;

    Block value(tsize);
    Tiler tiler(tsize);
    // std::cout << row_flag << " " << con_flag << " " << col_flag << std::endl;
    for (size_t it=0; it<ntrow1+row_flag; ++it)
    {
        size_t tile_row_size = (row_flag & (ntrow1 == it)) ? row_spare_size : tsize; 
        for (size_t kt=0; kt<ntcol2+col_flag; ++kt)
        {
            size_t tile_col_size = (col_flag & (ntcol2 == kt)) ? col_spare_size : tsize; 
            value = 0;
            for (size_t jt=0; jt<ntcol1+con_flag; ++jt)
            {
                size_t tile_con_size = (con_flag & (ntcol1 == jt)) ? con_spare_size : tsize; 
                tiler.load(
                    mat1, it, jt, tile_row_size, tile_con_size, 
                    mat2, jt, kt, tile_con_size, tile_col_size, true
                );
                tiler.AVX_multiply(tile_row_size, tile_col_size, tile_con_size);
                value += (*tiler.m_ret);

            }

            value.save(ret, it, kt, tile_row_size, tile_col_size);
        }
    }
    return ret;
}

// using Pthread

// def arg
typedef struct
{   
    const Matrix *mat1;
    const Matrix *mat2;
    // Block *block;
    // Tiler *tile;
    int threadId;
    int numThreads;
    int num_block;
    int ch_thread;

    size_t ntrow1;
    size_t ntcol1;
    size_t ntcol2;

    size_t row_spare_size;
    size_t con_spare_size;
    size_t col_spare_size;

    size_t row_flag;
    size_t con_flag;
    size_t col_flag;

    size_t tsize;
    Matrix *ret;
} WorkerArg;

void* threadstarts_backup(void *arg)
{   
    // clock_t s1,e1;
    // clock_t s2,e2;

    // s: whick block
    WorkerArg *args = (WorkerArg *)arg;
    Block value(args->tsize);
    Tiler tiler(args->tsize);
    // int cnt = 0;
    // double avg_load = 0.0;
    // s1 = clock(); 
    for (int t_id = args->threadId; t_id < args->num_block; t_id+=args->numThreads){
        size_t it = t_id / args->ntrow1;
        size_t kt = t_id % args->ntrow1;
        size_t tile_row_size = (args->row_flag & (args->ntrow1 == it)) ? args->row_spare_size : args->tsize;
        size_t tile_col_size = (args->col_flag & (args->ntcol2 == kt)) ? args->col_spare_size : args->tsize;
        // s2 = clock();
        value = 0;
        for (size_t jt = 0; jt < args->ntcol1 + args->con_flag; ++jt){
            size_t tile_con_size = (args->con_flag & (args->ntcol1 == jt)) ? args->con_spare_size : args->tsize;
            tiler.load(
                        (*args->mat1), it, jt, tile_row_size, tile_con_size, 
                        (*args->mat2), jt, kt, tile_con_size, tile_col_size
                    );
            tiler.multiply(tile_row_size, tile_col_size, tile_con_size);
            value += (*tiler.m_ret);
        }    
        value.save(*args->ret, it, kt, tile_row_size, tile_col_size); 
        // e2 = clock();
        // avg_load += (double)(e2-s2) / CLOCKS_PER_SEC;
        // cnt++;
    }
    // e1 = clock();
    // double diff = (double)(e1-s1) / CLOCKS_PER_SEC;
    // printf("Thread %2d : time forloop threads = %f avg work load = %f \n", args->threadId, diff, avg_load/cnt);
    pthread_exit((void *)0);
}

void* threadstarts(void *arg)
{   
    clock_t s1,e1;
    clock_t s2,e2;

    WorkerArg *args = (WorkerArg *)arg;
    // Case 1
    Block value(args->tsize);
    Tiler tile(args->tsize);
    // Case 2
    // Block &value = *args->block;
    // Tiler &tile = *args->tile;

    int cnt = 0;
    double avg_load = 0.0;
    s1 = clock(); 
    //////////////////////////////////////////////
    size_t unit_load = size_t((args->ntrow1+args->row_flag) / args->numThreads);
    size_t st_load = unit_load * args->threadId;
    size_t end_load = args->threadId != args->numThreads-1 ? unit_load * (1+args->threadId) : (args->ntrow1+args->row_flag);

    if (args->threadId < args->ch_thread)
    for (size_t it = st_load; it < end_load; ++it)
    {
        size_t tile_row_size = (args->row_flag & (args->ntrow1 == it)) ? args->row_spare_size : args->tsize; 
        for (size_t kt=0; kt<args->ntcol2+args->col_flag; ++kt)
        {
            size_t tile_col_size = (args->col_flag & (args->ntcol2 == kt)) ? args->col_spare_size : args->tsize; 
            s2 = clock();
            value = 0;
            for (size_t jt = 0; jt < args->ntcol1 + args->con_flag; ++jt){
                size_t tile_con_size = (args->con_flag & (args->ntcol1 == jt)) ? args->con_spare_size : args->tsize;
                tile.load(
                            (*args->mat1), it, jt, tile_row_size, tile_con_size, 
                            (*args->mat2), jt, kt, tile_con_size, tile_col_size
                        );
                tile.multiply(tile_row_size, tile_col_size, tile_con_size);
                value += (*tile.m_ret);
            }    
            value.save(*args->ret, it, kt, tile_row_size, tile_col_size); 
            e2 = clock();
            avg_load += (double)(e2-s2) / CLOCKS_PER_SEC;
            cnt++;
        }
    }
    e1 = clock();
    double diff = (double)(e1-s1) / CLOCKS_PER_SEC;
    // printf("Thread %2d : time forloop threads = %f | avg work load = %f cnt = %d\n", args->threadId, diff, avg_load/cnt, cnt);
    pthread_exit((void *)0);
}

Matrix multiply_tile_modify_pthread(const Matrix &mat1, const Matrix &mat2, size_t block_size, int numThreads, int ch_thread)
{   
    Matrix ret(mat1.nrow(), mat2.ncol());

    const size_t tsize = block_size;

    const size_t nrow1 = mat1.nrow();
    const size_t ncol1 = mat1.ncol();
    // const size_t nrow2 = mat2.nrow();
    const size_t ncol2 = mat2.ncol();

    const size_t ntrow1 = nrow1 / tsize;
    const size_t ntcol1 = ncol1 / tsize;
    // const size_t ntrow2 = nrow2 / tsize;
    const size_t ntcol2 = ncol2 / tsize;

    const size_t row_spare_size = nrow1 % tsize;
    const size_t con_spare_size = ncol1 % tsize;
    const size_t col_spare_size = ncol2 % tsize;

    const size_t row_flag = row_spare_size > 0;
    const size_t con_flag = con_spare_size > 0;
    const size_t col_flag = col_spare_size > 0;

    constexpr const int MAX_THREADS = 12;
    pthread_t threads[MAX_THREADS];
    WorkerArg args[MAX_THREADS];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for (int i = 0; i < numThreads; i++){
        args[i].mat1 = &mat1;
        args[i].mat2 = &mat2;
        // args[i].tile = new Tiler(tsize);
        // args[i].block = new Block(tsize);


        args[i].threadId = i;
        args[i].numThreads = numThreads;
        args[i].num_block = (ntrow1+row_flag) * (ntcol2+col_flag);
        args[i].ch_thread = ch_thread;

        args[i].ntrow1 = ntrow1;
        args[i].ntcol1 = ntcol1;
        args[i].ntcol2 = ntcol2;

        args[i].row_spare_size = row_spare_size;
        args[i].con_spare_size = con_spare_size;
        args[i].col_spare_size = col_spare_size;

        args[i].row_flag = row_flag;
        args[i].con_flag = con_flag;
        args[i].col_flag = col_flag;
        
        args[i].tsize = tsize;
        // Case 1
        args[i].ret = &ret;
        // Case 2
        // args[i].ret = new Matrix(mat1.nrow(), mat2.ncol());
    }

    for (int i = 0; i < numThreads; i++) pthread_create(&threads[i], NULL, &threadstarts, (void *)&args[i]);
    for (int i = 0; i < numThreads; i++) pthread_join(threads[i], NULL);
    
    pthread_attr_destroy(&attr);
    return ret;
}

void* naive_threadstarts(void *arg)
{
    WorkerArg *args = (WorkerArg*)arg;
    size_t row = args->mat1->nrow();
    size_t col = args->mat2->ncol();
    size_t content = args->mat1->ncol();
    size_t load = row / args->numThreads;
    size_t start = args->threadId * load;
    size_t end = args->threadId == args->numThreads-1 ? row : start + load ;
    for (size_t i=start; i<end; i++) {
        for (size_t j=0; j<col; j++) {
            double sum=0.0;
            for (size_t k=0; k<content; k++) {
                sum += args->mat1->operator()(i,k) * args->mat2->operator()(k,j);
            }
            args->ret->operator()(i,j)=sum;
        }
    }
    pthread_exit((void *)0);
}

Matrix multiply_naive_pthread(const Matrix &mat1, const Matrix &mat2, int numThreads)
{
    Matrix ret(mat1.nrow(), mat2.ncol());

    constexpr const int MAX_THREADS = 12;
    pthread_t threads[MAX_THREADS];
    WorkerArg args[MAX_THREADS];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for (int i = 0; i < numThreads; i++){
        args[i].mat1 = &mat1;
        args[i].mat2 = &mat2;

        args[i].threadId = i;
        args[i].numThreads = numThreads;

        // Case 1
        args[i].ret = &ret;
        // Case 2
        // args[i].ret = new Matrix(mat1.nrow(), mat2.ncol());
    }

    for (int i = 0; i < numThreads; i++) pthread_create(&threads[i], NULL, &naive_threadstarts, (void *)&args[i]);
    for (int i = 0; i < numThreads; i++) pthread_join(threads[i], NULL);
    
    pthread_attr_destroy(&attr);
    return ret;
}

Matrix multiply_naive_omp(const Matrix &mat1, const Matrix &mat2, int numThreads) 
{
    omp_set_num_threads(numThreads);
    size_t row=mat1.nrow();
    size_t col=mat2.ncol();
    size_t content=mat1.ncol();
    Matrix tmp(row, col);

    size_t i,j,k;
    #pragma omp parallel for private(i,j,k) shared(mat1,mat2,tmp)
    for ( i=0; i<row; i++) {
        for ( j=0; j<col; j++) {
            double sum=0.0;
            for ( k=0; k<content; k++) {
                sum+=mat1(i,k)*mat2(k,j);
            }
            tmp(i,j)=sum;
        }
    }
    return tmp;
}





void SetMatrixMode(int val)
{
    if (val > 0)
        Matrix::multiplication_mode = val;
    else
        throw std::runtime_error("Matrix::multiplication_mode should  greater than 0 !!\n");

}

int GetMatrixMode()
{
    return Matrix::multiplication_mode;
}

