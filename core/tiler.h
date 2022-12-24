#pragma once
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <functional>
#include "matrix.h"
// template<size_t N>
struct Block
{
    // const size_t NDIM = N;

    double   operator[] (size_t idx) const { return m_buffer[idx]; }
    double & operator[] (size_t idx)       { return m_buffer[idx]; }
    Block(size_t v): N(v), NDIM(v){
        m_buffer=new double[N*N];
    }
    ~Block() {
        delete m_buffer;
    }

    Block & operator= (double v)
    {
        for (size_t i=0; i<N*N; ++i) { m_buffer[i] = v; }
        return *this;
    }

    Block & operator+= (Block const & other)
    {
        for (size_t i=0; i<N*N; ++i) { m_buffer[i] += other.m_buffer[i]; }
        return *this;
    }

    void save(Matrix & mat, size_t it, size_t jt);
    void save(Matrix & mat, size_t it, size_t jt, size_t nrow, size_t ncol);

    size_t N;
    size_t NDIM;
    double *m_buffer = NULL;
};

// template<size_t N> 
void Block::save(
    Matrix & mat, size_t it, size_t jt
)
{
    const size_t ncol = mat.ncol();

    for (size_t i=0; i<NDIM; ++i)
    {
        const size_t base_s = i*NDIM;
        const size_t base_t = (it*NDIM + i) * ncol + jt*NDIM;

        for (size_t j=0; j<NDIM; ++j)
        {
            mat.m_buffer[base_t + j] = m_buffer[base_s + j];
        }
    }
}

void Block::save(
    Matrix & mat, size_t it, size_t jt, size_t p_nrow, size_t p_ncol
)
{
    const size_t ncol = mat.ncol();

    for (size_t i=0; i<p_nrow; ++i)
    {
        const size_t base_s = i*NDIM;
        const size_t base_t = (it*NDIM + i) * ncol + jt*NDIM;

        for (size_t j=0; j<p_ncol; ++j)
        {
            mat.m_buffer[base_t + j] = m_buffer[base_s + j];
        }
    }
}


// template<size_t N>
struct Tiler
{
    Tiler(size_t v) : NDIM(v), N(v) {
        m_mat1 = new Block(v);
        m_mat2 = new Block(v);
        m_ret = new Block(v);
    }
    ~Tiler() {
        delete m_mat1;
        delete m_mat2;
        delete m_ret;
    }
    // static constexpr const size_t NDIM = N;
    size_t NDIM;
    size_t N;

    void load(
        Matrix const & mat1, size_t it1, size_t jt1,
        Matrix const & mat2, size_t it2, size_t jt2
    );
    void load(
        Matrix const & mat1, size_t it1, size_t jt1, size_t it1_size, size_t jt1_size,
        Matrix const & mat2, size_t it2, size_t jt2, size_t it2_size, size_t jt2_size
    );

    void multiply();
    void multiply(size_t res_row, size_t res_col, size_t res_con);

    Block *m_mat1 = NULL; // row-major
    size_t it1_size, ji1_size;
    Block *m_mat2 = NULL; // column-major
    size_t it2_size, ji2_size;
    Block *m_ret = NULL; // row-major
    size_t it3_size, ji3_size;
};

void Tiler::load(
    Matrix const & mat1, size_t it1, size_t jt1,
    Matrix const & mat2, size_t it2, size_t jt2
)
{
    const size_t ncol1 = mat1.ncol();
    // std::cout << "section 1" << std::endl;

    for (size_t i=0; i<NDIM; ++i)
    {
        const size_t base_t = i*NDIM;
        const size_t base_s = (it1*NDIM + i) * ncol1 + jt1*NDIM;

        for (size_t j=0; j<NDIM; ++j)
        {
            (*m_mat1)[base_t + j] = mat1.m_buffer[base_s + j];
            //std::cout << (*m_mat1)[base_t + j] << " ";
        }
    }
    // std::cout << "section 2" << std::endl;

    const size_t ncol2 = mat2.ncol();

    for (size_t i=0; i<NDIM; ++i)
    {
        const size_t base_t = i*NDIM;
        const size_t base_s = (it2*NDIM + i) * ncol2 + jt2*NDIM;

        for (size_t j=0; j<NDIM; ++j)
        {
            // std::cout << "idx: " << base_t+j << "addr: " << m_ret->m_buffer <<std::endl;
            (*m_ret)[base_t + j] = mat2.m_buffer[base_s + j];

            (*m_mat2)[j*NDIM + i] = (*m_ret)[base_t + j];
            //std::cout <<  (*m_mat2)[j*NDIM + i] << " ";
        }
    }
}

void Tiler::load(
    Matrix const & mat1, size_t it1, size_t jt1, size_t it1_size, size_t jt1_size,
    Matrix const & mat2, size_t it2, size_t jt2, size_t it2_size, size_t jt2_size
)
{
    const size_t ncol1 = mat1.ncol();

    for (size_t i=0; i<it1_size; ++i)
    {
        const size_t base_t = i*NDIM;
        const size_t base_s = (it1*NDIM + i) * ncol1 + jt1*NDIM;

        for (size_t j=0; j<jt1_size; ++j)
        {
            (*m_mat1)[base_t + j] = mat1.m_buffer[base_s + j];
            //std::cout << (*m_mat1)[base_t + j] << " ";
        }
        //std::cout << std::endl;
    }
    //std::cout << std::endl;
    // std::cout << "section 2" << std::endl;

    const size_t ncol2 = mat2.ncol();

    for (size_t i=0; i<it2_size; ++i)
    {
        const size_t base_t = i*NDIM;
        const size_t base_s = (it2*NDIM + i) * ncol2 + jt2*NDIM;

        for (size_t j=0; j<jt2_size; ++j)
        {
            (*m_ret)[base_t + j] = mat2.m_buffer[base_s + j];
            (*m_mat2)[j*NDIM + i] = (*m_ret)[base_t + j];
            //std::cout <<  (*m_mat2)[j*NDIM + i] << " ";
        }
        //std::cout << std::endl;
    }
    //std::cout << std::endl;
}


void Tiler::multiply()
{
    for (size_t i=0; i<NDIM; ++i)
    {
        const size_t base1 = i*NDIM;

        for (size_t k=0; k<NDIM; ++k)
        {
            const size_t base2 = k*NDIM;

            double v = 0;
            for (size_t j=0; j<NDIM; ++j)
            {
                v += (*m_mat1)[base1 + j] * (*m_mat2)[base2 + j];
            }
            (*m_ret)[base1 + k] = v;
        }
    }
}

void Tiler::multiply(size_t res_row, size_t res_col, size_t res_con)
{
    for (size_t i=0; i<res_row; ++i)
    {
        const size_t base1 = i*NDIM;

        for (size_t k=0; k<res_col; ++k)
        {
            const size_t base2 = k*NDIM;

            double v = 0;
            for (size_t j=0; j<res_con; ++j)
            {
                v += (*m_mat1)[base1 + j] * (*m_mat2)[base2 + j];
            }
            (*m_ret)[base1 + k] = v;
            //std::cout << v << " ";
        }
        //std::cout << std::endl;
    }
}