#include "stdio.h"
#include "linear.h"
#include "matrix.h"


Linear::Linear(int in_feat, int out_feat, bool use_bias, bool trainable):
    BaseLayer(trainable, true), m_in_feat(in_feat), m_out_feat(out_feat), m_use_bias(use_bias)
{
    // # TODO random initialize matrix
    m_weight = Matrix(in_feat, out_feat);
    // std::cout << "create m_weight: " << m_weight.get_buffer() << std::endl;
    if (use_bias) {
        m_bias = Matrix(1, out_feat);
    }
}

Linear::~Linear() {
    // printf("calling layer destructor \n");
}

// input_tensor=(batch, feat_dim)
Matrix Linear::forward(const Matrix &input_tensor)
{
    if (input_tensor.ncol() != m_weight.nrow()) throw std::runtime_error("mis match input_tensor.ncol and m_weight.nrow !\n");
    // m_weight=(feat_dim, out_dim)
    // output=(batch, out_dim)
    Matrix output = multiply_mkl(input_tensor, m_weight);
    if (m_use_bias) {
        // TODO broadcast m_bias
        output = output + m_bias;
    }
    return output;
}

// gradient=(batch, out_feat)
std::pair<Matrix,pybind11::tuple> Linear::backward(Matrix &gradient)
{
    printf("*************************************************\n");
    // m_input=(batch, in_feat)
    // gradient=(batch, out_feat)
    // m_weight_gradient=(in_feat, out_feat)
    m_weight_gradient=multiply_mkl(m_input.T(), gradient);
    std::cout << "m_input.T(): ";
    m_input.T().print_shape();
    std::cout << "gradient_flow: ";
    gradient.print_shape();
    std::cout << "m_weight_gradient: ";
    m_weight_gradient.print_shape();
    // m_weight=(in_feat, out_feat)
    // m_weight.T=(out_feat, in_feat)
    // graident=(batch, out_feat)
    // dzda=(batch, in_feat)
    Matrix dzda = multiply_mkl(gradient, m_weight.T());
    printf("*************************************************\n");
    if (m_use_bias)
    {
        Matrix ones=Matrix::fillwith(1, gradient.nrow(), 1.0);
        m_bias_gradient = multiply_mkl(ones, gradient);
        return std::pair<Matrix, pybind11::tuple>(
            dzda, 
            pybind11::make_tuple(
                m_weight_gradient, 
                m_bias_gradient
            )
        );
    }

    return std::pair<Matrix, pybind11::tuple>(
            dzda, 
            pybind11::make_tuple(m_weight_gradient)
        );
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


