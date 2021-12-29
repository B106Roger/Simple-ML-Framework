#include "stdio.h"
#include "linear.h"
#include "matrix.h"


Linear::Linear(int in_feat, int out_feat, bool use_bias, bool trainable):
    BaseLayer(trainable, true), m_in_feat(in_feat), m_out_feat(out_feat), m_use_bias(use_bias)
{
    // # TODO random initialize matrix
    m_weight = Matrix(out_feat, in_feat);
    // std::cout << "create m_weight: " << m_weight.get_buffer() << std::endl;
    if (use_bias) {
        m_bias = Matrix(out_feat, 1);
    }
}

Linear::~Linear() {
    // printf("calling layer destructor \n");
}

// input_tensor=(feat_dim, batch)
Matrix Linear::forward(const Matrix &input_tensor)
{
    if (input_tensor.ncol() != m_weight.nrow()) throw std::runtime_error("mis match input_tensor.ncol and m_weight.nrow !\n");
    // m_weight=(out_dim, feat_dim)
    // output=(out_dim, batch)
    Matrix output = multiply_mkl(input_tensor, m_weight);
    // output.print_shape();
    // input_tensor.print_shape();
    // m_weight.print_shape();
    if (m_use_bias) {
        // TODO broadcast m_bias
        output = output + m_bias;
    }
    return output;
}

void Linear::set_weight(pybind11::tuple py_tuple)
{
    // check weight is compatible
    Matrix weight = py_tuple[0].cast<Matrix>();
    if (weight.nrow() != m_in_feat || weight.ncol() != m_out_feat) 
        throw std::runtime_error("invalid weight shape for layer to consume");
    // assign weight
    m_weight = weight;

    if (m_use_bias) {
        Matrix bias = py_tuple[1].cast<Matrix>();
        // check bias is compatible
        if (bias.nrow() != 1 || bias.ncol() != m_out_feat) 
            throw std::runtime_error("invalid bias shape for layer to consume");
        // assign bias
        m_bias = bias;
    }
}


