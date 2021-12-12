#include <pybind11/pybind11.h>
#include <iostream>
#include <math.h>
#include <mkl.h>
#include "matrix.h"

namespace py = pybind11;

class Block {
public:
    Block(size_t nrow, size_t ncol, bool colmajor):
        m_nrow(nrow), m_ncol(ncol), m_buffer(NULL), m_colmajor(colmajor)
    {
        if (m_colmajor)
            m_buffer=new double[m_nrow*m_ncol];
    }
    Block(const Block &block):
        m_nrow(block.m_nrow), m_ncol(block.m_ncol), m_buffer(NULL), m_colmajor(block.m_colmajor)
    {
        if (block.m_colmajor)
        {
            m_buffer=new double[m_nrow*m_ncol];
            memcpy(m_buffer, block.m_buffer, sizeof(double) * m_nrow * m_ncol);
        }
    }
    ~Block() { 
        if (m_colmajor) delete[] m_buffer;
        m_buffer = NULL;
    }
    double   operator() (size_t row, size_t col) const { // for getitem
        // if (row > m_nrow) throw std::invalid_argument( "received row exceed nrow" );
        // if (col > m_ncol) throw std::invalid_argument( "received col exceed ncol" );
        if (m_colmajor)
        {
            return m_buffer[col * m_nrow + row];
        }
        else
            return m_buffer[row * m_row_stride + col];
    }
    void setContent(double *ptr, size_t row_stride) {
        m_row_stride = row_stride;
        if (m_colmajor) {
            for (int i = 0; i < m_nrow; i++) {
                for (int j = 0; j < m_ncol; j++) {
                    m_buffer[j * m_nrow + i]= ptr[i * m_row_stride + j];
                }
            }
        } else {
            
            m_buffer = ptr;
        }
    }

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

    Matrix(size_t nrow, size_t ncol)
      : m_nrow(nrow), m_ncol(ncol)
    {
        size_t nelement = nrow * ncol;
        m_buffer = new double[nelement];
        memset(m_buffer, 0, nelement*sizeof(double));
    }

    Matrix(Matrix const &target) {
        int row=target.nrow();
        int col=target.ncol();
        m_buffer = new double[row*col];
        m_nrow=row;
        m_ncol=col;
        memcpy(m_buffer, target.m_buffer, sizeof(double) * row * col);
    }

    ~Matrix() { delete[] m_buffer;}

    // No bound check.
    double   operator() (size_t row, size_t col) const { // for getitem
        // if (row > m_nrow) throw std::invalid_argument( "received row exceed nrow" );
        // if (col > m_ncol) throw std::invalid_argument( "received col exceed ncol" );
        return m_buffer[row*m_ncol + col];
    }
    double & operator() (size_t row, size_t col) {       // for setitem
        // if (row > m_nrow) throw std::invalid_argument( "received row exceed nrow" );
        // if (col > m_ncol) throw std::invalid_argument( "received col exceed ncol" );
        return m_buffer[row*m_ncol + col];
    }
    Matrix operator+(const Matrix &mat) const {
        // if (m_nrow != mat.m_nrow || m_ncol != mat.m_ncol) throw std::invalid_argument( "received matrix shape isn't consistent" );
        Matrix result(mat);
        for (int i=0; i< m_nrow; i+=1) {
            for (int j=0; j<m_ncol; j+=1) {
                result.m_buffer[i*m_ncol+j]+=(*this)(i,j);
            }
        }
        return result;
    }
    void operator+=(const Matrix &mat) {
        // if (m_nrow != mat.m_nrow || m_ncol != mat.m_ncol) throw std::invalid_argument( "received matrix shape isn't consistent" );
        for (int i=0; i< m_nrow; i+=1) {
            for (int j=0; j<m_ncol; j+=1) {
                m_buffer[i*m_ncol+j]+=mat(i,j);
            }
        }
    }
    Matrix operator-(const Matrix &mat) const {
        // if (m_nrow != mat.m_nrow || m_ncol != mat.m_ncol) throw std::invalid_argument( "received matrix shape isn't consistent" );
        Matrix result(mat);
        for (int i=0; i< m_nrow; i+=1) {
            for (int j=0; j<m_ncol; j+=1) {
                result.m_buffer[i*m_ncol+j]-=(*this)(i,j);
            }
        }
        return result;
    }
    void operator-=(const Matrix &mat) {
        // if (m_nrow != mat.m_nrow || m_ncol != mat.m_ncol) throw std::invalid_argument( "received matrix shape isn't consistent" );
        for (int i=0; i< m_nrow; i+=1) {
            for (int j=0; j<m_ncol; j+=1) {
                m_buffer[i*m_ncol+j]-=mat(i,j);
            }
        }
    }
    void operator=(const Matrix &target) {
        delete[] m_buffer;
        int row=target.nrow();
        int col=target.ncol();
        m_buffer = new double[row*col];
        m_nrow=row;
        m_ncol=col;
        memcpy(m_buffer, target.m_buffer, sizeof(double) * row * col);
    }
    bool operator==(const Matrix &target) const{
        if (m_nrow != target.m_nrow || m_ncol != target.m_ncol) {
            return false;
        } else {
            for (int i = 0; i < m_nrow; i++) {
                for (int j = 0; j < m_ncol; j++) {
                    if ((*this)(i,j) != target(i,j)) return false;
                }
            }
            return true;
        }
    }

    Block get_block(size_t block_size, size_t row_idx, size_t col_idx, bool col2row = false) const{
        // row_idx: row index of the block
        // col_idx: col index of the block
        // if (block_size*row_idx > m_nrow) throw std::invalid_argument( "received block_size*row exceed nrow" );
        // if (block_size*col_idx > m_ncol) throw std::invalid_argument( "received block_size*col exceed ncol" );
        size_t bk_col = m_ncol - block_size*col_idx < block_size ? m_ncol - block_size*col_idx : block_size;
        size_t bk_row = m_nrow - block_size*row_idx < block_size ? m_nrow - block_size*row_idx : block_size;
        Block block(bk_row, bk_col, col2row);

        size_t target_row=(block_size*row_idx)*m_ncol;
        size_t target_col=(block_size*col_idx);
        block.setContent(m_buffer+target_row+target_col, m_ncol);
        return block;
    }

    void set_block(size_t block_size, size_t row_idx, size_t col_idx, const Matrix &mat) {
        // row_idx: row index of the block
        // col_idx: col index of the block
        // if (block_size*row_idx > m_nrow || block_size*col_idx > m_ncol) throw std::invalid_argument( "received block_size*row exceed nrow or block_size*col exceed ncol" );
        size_t bk_col = m_ncol - block_size*col_idx < block_size ? m_ncol - block_size*col_idx : block_size;
        size_t bk_row = m_nrow - block_size*row_idx < block_size ? m_nrow - block_size*row_idx : block_size;
        // if (bk_row != mat.m_nrow || bk_col != mat.m_ncol) throw std::invalid_argument( "received block_shape and mat are not consistent" );

        for (int i=0;i<bk_row; i++) {
            size_t target_row=(block_size*row_idx+i)*m_ncol;
            size_t target_col=(block_size*col_idx);
            size_t source_row=i*bk_col;
            memcpy(m_buffer+target_row+target_col, mat.m_buffer+source_row, sizeof(double) * mat.m_ncol);
        }
    }

    double *data() { return m_buffer; }
    size_t nrow() const { return m_nrow; }
    size_t ncol() const { return m_ncol; }
    friend Matrix multiply_mkl(Matrix &mat1, Matrix &mat2);

private:
    
    size_t m_nrow;
    size_t m_ncol;
    double * m_buffer;

};

Matrix multiply_naive_bk(const Block &mat1, const Block &mat2) {
    size_t row=mat1.nrow();
    size_t col=mat2.ncol();
    size_t content=mat1.ncol();
    Matrix tmp(row, col);
    for (int i=0; i<row; i++) {
        for (int j=0; j<col; j++) {
            double sum=0.0;
            for (int k=0; k<content; k++) {
                sum+=mat1(i,k)*mat2(k,j);
            }
            tmp(i,j)=sum;
        }
    }
    return tmp;
}

Matrix multiply_naive(const Matrix &mat1, const Matrix &mat2) {
    size_t row=mat1.nrow();
    size_t col=mat2.ncol();
    size_t content=mat1.ncol();
    Matrix tmp(row, col);
    for (int i=0; i<row; i++) {
        for (int j=0; j<col; j++) {
            double sum=0.0;
            for (int k=0; k<content; k++) {
                sum+=mat1(i,k)*mat2(k,j);
            }
            tmp(i,j)=sum;
        }
    }
    return tmp;
}

Matrix multiply_tile(Matrix &mat1, Matrix &mat2, size_t block_size) {
    size_t row=mat1.nrow();
    size_t col=mat2.ncol();
    size_t content=mat1.ncol();
    Matrix result(row, col);
    int max_bk_row = row % block_size == 0 ? row/block_size : row/block_size+1;
    int max_bk_col = col % block_size == 0 ? col/block_size : col/block_size+1;
    int max_bk_content = content % block_size == 0 ? content/block_size : content/block_size+1;

    for (int i=0; i<max_bk_row; i++) {
        for (int j=0; j<max_bk_col; j++) {
            Matrix tmpmat(1,1);
            for (int k=0; k<max_bk_content; k++) {
                if (k==0) 
                    tmpmat = multiply_naive_bk(mat1.get_block(block_size, i, k, false), mat2.get_block(block_size, k, j, true));
                else
                    tmpmat +=  multiply_naive_bk(mat1.get_block(block_size, i, k, false), mat2.get_block(block_size, k, j, true));
            }
            result.set_block(block_size, i, j, tmpmat);
        }
    }
    return result;
}


Matrix multiply_mkl(Matrix &mat1, Matrix &mat2) {
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


PYBIND11_MODULE(_matrix, m) {
    m.doc() = "nsd21au hw3 pybind implementation"; // optional module docstring
    py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
        .def_buffer([](Matrix &m) -> py::buffer_info{
            return py::buffer_info(
                m.data(),
                sizeof(double),
                py::format_descriptor<double>::format(),
                2,
                {m.nrow(), m.ncol()},
                {sizeof(double) * m.ncol(), sizeof(double)}
            );
        })
        .def(py::init([](py::buffer b) {
            typedef Eigen::Stride<Eigen::Dynamic, EIgen::Dynamic> Strides;

        }))
        .def(pybind11::init<int,int>())
        .def("__setitem__", [](Matrix &mat, std::pair<size_t, size_t> idx, double val) { return mat(idx.first, idx.second) = val; })
        .def("__getitem__", [](const Matrix &mat, std::pair<size_t, size_t> idx) { return mat(idx.first, idx.second); })
        .def("__eq__", [](const Matrix &mat1, const Matrix &mat2) { return mat1 == mat2; })
        .def_property_readonly("nrow", &Matrix::nrow)
        .def_property_readonly("ncol", &Matrix::ncol);

    m.def("multiply_naive", &multiply_naive);
    m.def("multiply_tile", &multiply_tile);
    m.def("multiply_mkl", &multiply_mkl);
}