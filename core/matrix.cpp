#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <math.h>
#include <mkl.h>
#include "matrix.h"

namespace py = pybind11;
    ////////////////////////////////////////////////////////////////////////////////
    /////////////       Class Member for Block Class                     //////////
    ////////////////////////////////////////////////////////////////////////////////

    Block::Block(size_t nrow, size_t ncol, bool colmajor):
        m_nrow(nrow), m_ncol(ncol), m_buffer(NULL), m_row_stride(0), m_colmajor(colmajor)
    {
        if (m_colmajor)
            m_buffer=new double[m_nrow*m_ncol];
    }
    Block::Block(const Block &block):
        m_nrow(block.m_nrow), m_ncol(block.m_ncol), m_buffer(NULL), m_row_stride(0), m_colmajor(block.m_colmajor)
    {
        if (block.m_colmajor)
        {
            m_buffer=new double[m_nrow*m_ncol];
            memcpy(m_buffer, block.m_buffer, sizeof(double) * m_nrow * m_ncol);
        }
    }
    Block::~Block() { 
        if (m_colmajor) delete[] m_buffer;
        m_buffer = NULL;
    }
    double   Block::operator() (size_t row, size_t col) const { // for getitem
        if (m_colmajor)
        {
            return m_buffer[col * m_nrow + row];
        }
        else
            return m_buffer[row * m_row_stride + col];
    }
    void Block::setContent(double *ptr, size_t row_stride) {
        m_row_stride = row_stride;
        if (m_colmajor) {
            for (size_t i = 0; i < m_nrow; i++) {
                for (size_t j = 0; j < m_ncol; j++) {
                    m_buffer[j * m_nrow + i]= ptr[i * m_row_stride + j];
                }
            }
        } else {
            
            m_buffer = ptr;
        }
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    /////////////       Class Member for Matrix Class                     //////////
    ////////////////////////////////////////////////////////////////////////////////

    Matrix::Matrix()
        : m_nrow(0), m_ncol(0), m_buffer(NULL)
    {
    }

    Matrix::Matrix(size_t nrow, size_t ncol)
      : m_nrow(nrow), m_ncol(ncol), m_buffer(NULL)
    {
        size_t nelement = nrow * ncol;
        m_buffer = new double[nelement];
        memset(m_buffer, 0, nelement*sizeof(double));
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

    Matrix::Matrix(Matrix const &target) {
        int row=target.nrow();
        int col=target.ncol();
        m_buffer = new double[row*col];
        m_nrow=row;
        m_ncol=col;
        memcpy(m_buffer, target.m_buffer, sizeof(double) * row * col);
    }

    Matrix::~Matrix() { delete[] m_buffer;}

    // No bound check.
    double   Matrix::operator() (size_t row, size_t col) const { // for getitem
        return m_buffer[row*m_ncol + col];
    }
    double & Matrix::operator() (size_t row, size_t col) {       // for setitem
        return m_buffer[row*m_ncol + col];
    }
    // implement in vectorize mode
    Matrix Matrix::operator+(const Matrix &mat) const {
        size_t row=std::max(mat.m_nrow, m_nrow);
        size_t col=std::max(mat.m_ncol, m_ncol);
        
        // check broadcastable
        if (mat.m_nrow != m_nrow && !(mat.m_nrow == 1 || m_nrow == 1)) 
            throw std::runtime_error("The shape is not broadcastable in row.");
        else if (mat.m_ncol != m_ncol && !(mat.m_ncol == 1 || m_ncol == 1))
            throw std::runtime_error("The shape is not broadcastable in column.");

        Matrix result(row, col);
        for (size_t i = 0; i < row; i+=1) {
            for (size_t j = 0; j < col; j+=1) {
                result.m_buffer[i*col+j] = 
                    (*this)(i % m_nrow,     j % m_ncol) + 
                        mat(i % mat.m_nrow, j % mat.m_ncol);
            }
        }
        return result;
    }
    void Matrix::operator+=(const Matrix &mat) {
        for (size_t i=0; i< m_nrow; i+=1) {
            for (size_t j=0; j<m_ncol; j+=1) {
                m_buffer[i*m_ncol+j]+=mat(i,j);
            }
        }
    }
    Matrix Matrix::operator-(const Matrix &mat) const {
        Matrix result(mat);
        for (size_t i=0; i< m_nrow; i+=1) {
            for (size_t j=0; j<m_ncol; j+=1) {
                result.m_buffer[i*m_ncol+j]-=(*this)(i,j);
            }
        }
        return result;
    }
    void Matrix::operator-=(const Matrix &mat) {
        for (size_t i=0; i< m_nrow; i+=1) {
            for (size_t j=0; j<m_ncol; j+=1) {
                m_buffer[i*m_ncol+j]-=mat(i,j);
            }
        }
    }
    void Matrix::operator=(const Matrix &target) {
        delete[] m_buffer;
        int row=target.nrow();
        int col=target.ncol();
        m_buffer = new double[row*col];
        m_nrow=row;
        m_ncol=col;
        memcpy(m_buffer, target.m_buffer, sizeof(double) * row * col);
    }
    bool Matrix::operator==(const Matrix &target) const{
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

    Matrix Matrix::T()const
    {
        Matrix result(m_ncol, m_nrow);
        for(size_t i = 0; i<m_nrow*m_ncol; i++)
        {
            size_t row_idx = i / m_ncol;
            size_t col_idx = i % m_nrow;
            result(row_idx, col_idx) = m_buffer[i];
        }
        return result;
    }

    Block Matrix::get_block(size_t block_size, size_t row_idx, size_t col_idx, bool col2row) const{
        // row_idx: row index of the block
        // col_idx: col index of the block
        size_t bk_col = m_ncol - block_size*col_idx < block_size ? m_ncol - block_size*col_idx : block_size;
        size_t bk_row = m_nrow - block_size*row_idx < block_size ? m_nrow - block_size*row_idx : block_size;
        Block block(bk_row, bk_col, col2row);

        size_t target_row=(block_size*row_idx)*m_ncol;
        size_t target_col=(block_size*col_idx);
        block.setContent(m_buffer+target_row+target_col, m_ncol);
        return block;
    }

    void Matrix::set_block(size_t block_size, size_t row_idx, size_t col_idx, const Matrix &mat) {
        // row_idx: row index of the block
        // col_idx: col index of the block
        size_t bk_col = m_ncol - block_size*col_idx < block_size ? m_ncol - block_size*col_idx : block_size;
        size_t bk_row = m_nrow - block_size*row_idx < block_size ? m_nrow - block_size*row_idx : block_size;
        for (size_t i=0;i<bk_row; i++) {
            size_t target_row=(block_size*row_idx+i)*m_ncol;
            size_t target_col=(block_size*col_idx);
            size_t source_row=i*bk_col;
            memcpy(m_buffer+target_row+target_col, mat.m_buffer+source_row, sizeof(double) * mat.m_ncol);
        }
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



Matrix multiply_naive_bk(const Block &mat1, const Block &mat2) {
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

Matrix multiply_naive(const Matrix &mat1, const Matrix &mat2) {
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

Matrix multiply_tile(Matrix &mat1, Matrix &mat2, size_t block_size) {
    size_t row=mat1.nrow();
    size_t col=mat2.ncol();
    size_t content=mat1.ncol();
    Matrix result(row, col);
    size_t max_bk_row = row % block_size == 0 ? row/block_size : row/block_size+1;
    size_t max_bk_col = col % block_size == 0 ? col/block_size : col/block_size+1;
    size_t max_bk_content = content % block_size == 0 ? content/block_size : content/block_size+1;

    for (size_t i=0; i<max_bk_row; i++) {
        for (size_t j=0; j<max_bk_col; j++) {
            Matrix tmpmat(1,1);
            for (size_t k=0; k<max_bk_content; k++) {
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

// Matrix multiply_numpy(py::array_t<double, py::array::c_style | py::array::forcecast> array, Matrix &mat2)
// {

// }

void test(py::buffer b) {
    py::buffer_info info = b.request();
    std::cout << info.format << std::endl;
}


PYBIND11_MODULE(_matrix, m) {
    m.doc() = "nsd21au hw3 pybind implementation"; // optional module docstring
    py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
        // given Matrix instance and convert to numpy object
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
        // given numpy array instance and convert to Matrix object
        .def(py::init([](py::buffer b)->Matrix {
            py::buffer_info info = b.request();
            if (info.ndim != 2)
                throw std::runtime_error("Incompatible buffer dimension!");
            size_t row = info.shape[0], col = info.shape[1];
            
            // Type Determine
            if (info.format == "L") // uint64
                return Matrix((unsigned int64_t*)info.ptr, row, col);
            else if (info.format == "l") // int64
                return Matrix((int64_t*)info.ptr, row, col);
            else if (info.format == "I") // uint32
                return Matrix((unsigned int32_t*)info.ptr, row, col);
            else if (info.format == "i") // int32
                return Matrix((int32_t*)info.ptr, row, col);
            else if (info.format == "H") // uint16
                return Matrix((unsigned int16_t*)info.ptr, row, col);
            else if (info.format == "h") // int16
                return Matrix((int16_t*)info.ptr, row, col);
            else if (info.format == "B") // uint8
                return Matrix((unsigned int8_t*)info.ptr, row, col);
            else if (info.format == "b") // int8
                return Matrix((int8_t*)info.ptr, row, col);
            else if (info.format == "f") // float32
                return Matrix((float*)info.ptr, row, col);
            else if (info.format == "d") // float64
                return Matrix((double*)info.ptr, row, col);
            else if (info.format == "g")  // float128
                return Matrix((long double*)info.ptr, row, col);
            else 
                // throw throw std::runtime_error("Incompatible buffer type!");
                return Matrix(1,1);
        }))
        .def(pybind11::init<int,int>())
        .def("__setitem__", [](Matrix &mat, std::pair<size_t, size_t> idx, double val) { return mat(idx.first, idx.second) = val; })
        .def("__getitem__", [](const Matrix &mat, std::pair<size_t, size_t> idx) { return mat(idx.first, idx.second); })
        .def("__add__", [](const Matrix &mat1, const Matrix &mat2) {return mat1 + mat2;})
        .def("__eq__", [](const Matrix &mat1, const Matrix &mat2) { return mat1 == mat2; })
        .def("T", &Matrix::T)
        .def_property_readonly("nrow", &Matrix::nrow)
        .def_property_readonly("ncol", &Matrix::ncol)
        .def_property("array", &Matrix::get_array, nullptr);
    m.def("multiply_naive", &multiply_naive);
    m.def("multiply_tile", &multiply_tile);
    m.def("multiply_mkl", &multiply_mkl);
    m.def("format_descriptor", &test);
}