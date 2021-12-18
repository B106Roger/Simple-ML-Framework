#include "linear.h"
#include "matrix.h"


Linear::Linear(int in_feat, int out_feat, bool use_bias, bool trainable):
    BaseLayer(trainable, true), m_in_feat(in_feat), m_out_feat(out_feat), m_use_bias(use_bias)
{
    // # TODO random initialize matrix
    m_weight = Matrix(out_feat, in_feat);
    if (use_bias) {
        m_bias = Matrix(out_feat, 1);
    }
}

Linear::~Linear() {}

// input_tensor=(feat_dim, batch)
Matrix Linear::forward(Matrix &input_tensor)
{
    // m_weight=(out_dim, feat_dim)
    // output=(out_dim, batch)
    Matrix output = multiply_mkl(m_weight, input_tensor); 
    if (m_use_bias) {
        // TODO broadcast m_bias
        output += m_bias;
    }
    return output;
}

void Linear::set_weight(pybind11::tuple py_tuple)
{
    // check weight is compatible
    Matrix weight = py_tuple[0].cast<Matrix>();
    if (weight.nrow() != m_out_feat or weight.ncol() != m_in_feat) 
        throw std::runtime_error("invalid weight shape for layer to consume");
    // assign weight
    m_weight = weight;


    if (m_use_bias) {
        Matrix bias = py_tuple[1].cast<Matrix>();
        // check bias is compatible
        if (bias.nrow() != m_out_feat or bias.ncol() != 1) 
            throw std::runtime_error("invalid bias shape for layer to consume");
        // assign bias
        m_bias = bias;
    }




}









// Matrix Linear::backward(Matrix &gradient)
// {
//     return ;
// }
