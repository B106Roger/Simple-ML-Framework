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
std::pair<Matrix,std::vector<Matrix>> Linear::backward(Matrix &gradient)
{
    // m_input=(batch, in_feat)
    // gradient=(batch, out_feat)
    // m_weight_gradient=(in_feat, out_feat)
    m_weight_gradient=multiply_mkl(m_input.T(), gradient);
    // m_weight=(in_feat, out_feat)
    // m_weight.T=(out_feat, in_feat)
    // graident=(batch, out_feat)
    // dzda=(batch, in_feat)
    Matrix dzda = multiply_mkl(gradient, m_weight.T());
    if (m_use_bias)
    {
        Matrix ones=Matrix::fillwith(1, gradient.nrow(), 1.0);
        m_bias_gradient = multiply_mkl(ones, gradient);
        return std::pair<Matrix, std::vector<Matrix>>(
            dzda, 
            {m_weight_gradient, m_bias_gradient}
        );
    }

    return std::pair<Matrix, std::vector<Matrix>>(
            dzda, 
            {m_weight_gradient}
        );
}

std::vector<Matrix> Linear::get_weight()
{
    if (m_use_bias) {
        return std::vector<Matrix>({m_weight, m_bias});
    }
    return std::vector<Matrix>({m_weight});

}

void Linear::set_weight(std::vector<Matrix> weight_list)
{
    // check weight is compatible
    Matrix &weight = weight_list[0];
    if (weight.nrow() != m_in_feat || weight.ncol() != m_out_feat) 
        throw std::runtime_error("invalid weight shape for layer to consume");
    // assign weight
    m_weight = weight;

    if (m_use_bias) {
        Matrix &bias = weight_list[1];
        // check bias is compatible
        if (bias.nrow() != 1 || bias.ncol() != m_out_feat) 
            throw std::runtime_error("invalid bias shape for layer to consume");
        // assign bias
        m_bias = bias;
    }
}

void Linear::apply_gradient(std::vector<Matrix> gradients)
{
    Matrix &w_grad = gradients[0];
    m_weight -= w_grad;
    if (m_use_bias) {
        Matrix &b_grad = gradients[1];
        m_bias -= b_grad;
    }
}

pybind11::dict Linear::get_config() const {
    return pybind11::dict(
        "in_dim"_a=m_in_feat,
        "out_dim"_a=m_out_feat,
        "use_bias"_a=m_use_bias,
        "has_var"_a=m_has_trainable_var
    );
}



