#pragma once
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <functional>
#include "matrix.h"
#include <immintrin.h>
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
        m_mat2_avx = new Block(v);
        m_ret = new Block(v);
    }
    ~Tiler() {
        delete m_mat1;
        delete m_mat2;
        delete m_mat2_avx;
        delete m_ret;
    }
    // static constexpr const size_t NDIM = N;
    size_t NDIM;
    size_t N;

    void load(
        Matrix const & mat1, size_t it1, size_t jt1,
        Matrix const & mat2, size_t it2, size_t jt2, bool use_avx=false
    );
    void load(
        Matrix const & mat1, size_t it1, size_t jt1, size_t it1_size, size_t jt1_size,
        Matrix const & mat2, size_t it2, size_t jt2, size_t it2_size, size_t jt2_size, bool use_avx=false
    );

    void multiply();
    void multiply(size_t res_row, size_t res_col, size_t res_con);
    void SSE_multiply(size_t res_row, size_t res_col, size_t res_con);
    void SSE_matmul(size_t m1_row, size_t m2_col);
    void AVX_multiply(size_t res_row, size_t res_col, size_t res_con);
    void AVX_matmul(size_t m1_row, size_t m2_col);

    Block *m_mat1 = NULL; // row-major
    size_t it1_size, ji1_size;
    Block *m_mat2 = NULL; // column-major
    Block *m_mat2_avx = NULL;
    size_t it2_size, ji2_size;
    Block *m_ret = NULL; // row-major
    size_t it3_size, ji3_size;
};

void Tiler::load(
    Matrix const & mat1, size_t it1, size_t jt1,
    Matrix const & mat2, size_t it2, size_t jt2, bool use_avx
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
            if (use_avx) (*m_mat2_avx)[base_t + j] = mat2.m_buffer[base_s + j];
            else (*m_mat2)[j*NDIM + i] = mat2.m_buffer[base_s + j];
        }
    }
}

void Tiler::load(
    Matrix const & mat1, size_t it1, size_t jt1, size_t it1_size, size_t jt1_size,
    Matrix const & mat2, size_t it2, size_t jt2, size_t it2_size, size_t jt2_size, bool use_avx
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
        }
    }
    const size_t ncol2 = mat2.ncol();

    for (size_t i=0; i<it2_size; ++i)
    {
        const size_t base_t = i*NDIM;
        const size_t base_s = (it2*NDIM + i) * ncol2 + jt2*NDIM;

        for (size_t j=0; j<jt2_size; ++j)
        {
            if (use_avx) (*m_mat2_avx)[base_t + j] = mat2.m_buffer[base_s + j];
            else (*m_mat2)[j*NDIM + i] = mat2.m_buffer[base_s + j];
        }
    }
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

void Tiler::SSE_multiply(size_t res_row, size_t res_col, size_t res_con)
{
    for (int i = 0; i < NDIM; i+=2)
        for (int j = 0; j < NDIM; j+=2)
            SSE_matmul(i, j);

}

void Tiler::SSE_matmul(size_t m1_row, size_t m2_col)
{
    __m128d I[2], R[2], T[2], S[2], Sum[2];

    for (int i = 0; i < 2; i++)
        Sum[i] = _mm_setzero_pd();

    for (int k = 0; k < NDIM; k += 2) {
        // Read matrix A
        for (int i = 0; i < 2; i++)
            I[i] = _mm_load_pd(&(*m_mat1)[(i+m1_row) * NDIM + k]);
        //double *I_v = (double *)&I[0];
        //std::cout << "I[0]:\t" << I_v[0] << "\t" << I_v[1];

        // Read matrix B
        for (int i = 0; i < 2; i++)
            R[i] = _mm_load_pd(&(*m_mat2)[(i+m2_col) * NDIM + k]);
        //double *R_v = (double *)&R[0];
        //std::cout << "R[0]:\t" << R_v[0] << "\t" << R_v[1];


        for (int i = 0; i < 2; i++) {
            // Inner product of vector from matrix A and B
            for (int j = 0; j < 2; j++)
                T[j] = _mm_mul_pd(I[i], R[j]);

            
            S[0] = _mm_unpacklo_pd(T[0], T[1]);
            S[1] = _mm_unpackhi_pd(T[0], T[1]);

            for (int j = 0; j < 2; j++)
                Sum[i] = _mm_add_pd(Sum[i], S[j]);
        }
    }

    for (int i = 0; i < 2; i++)
        _mm_store_pd(&(*m_ret)[(i+m1_row) * NDIM + m2_col], Sum[i]);
    //double *Sum_v = (double *)&Sum[0];
    //std::cout << "Sum[0]:\t" << Sum_v[0] << "\t" << Sum_v[1] << "\t" << Sum_v[2] << "\t" << Sum_v[3];
    //exit(0);
}


void Tiler::AVX_multiply(size_t res_row, size_t res_col, size_t res_con)
{
    for (int i = 0; i < NDIM; i+=4)
        for (int j = 0; j < NDIM; j+=4)
            AVX_matmul(i, j);

}

void Tiler::AVX_matmul(size_t m1_row, size_t m2_col)
{
    __m256d I[4], R[4], S[4], Sum[4];

    for (int i = 0; i < 4; i++)
        Sum[i] = _mm256_setzero_pd();

    for (int k = 0; k < NDIM; k += 4) {
        
        // Read matrix B
        for (int i = 0; i < 4; i++)
            R[i] = _mm256_loadu_pd(&(*m_mat2_avx)[(k + i) * NDIM + m2_col]); //k == m_mat1.col == m_mat2.row
        
        for (int i = 0; i < 4; i++) {
            // Read matrix A and then mul A and B
            for (int j = 0; j < 4; j++) {
                I[j] = _mm256_set1_pd((*m_mat1)[(m1_row + i) * NDIM + k + j]);
                S[j] = _mm256_mul_pd(R[j], I[j]);
            }

            for (int j = 0; j < 4; j++)
                Sum[i] = _mm256_add_pd(Sum[i], S[j]);
        }
    }

    for (int i = 0; i < 4; i++)
        _mm256_storeu_pd(&(*m_ret)[(i+m1_row) * NDIM + m2_col], Sum[i]);
    //double *Sum_v = (double *)&Sum[0];
    //std::cout << "Sum[0]:\t" << Sum_v[0] << "\t" << Sum_v[1] << "\t" << Sum_v[2] << "\t" << Sum_v[3];
    //exit(0);
}