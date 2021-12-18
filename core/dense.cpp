#include "dense.h"
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
    Matrix output = matrix_multiply(m_weight, input_tensor); 
    if (m_use_bias) {
        // TODO broadcast m_bias
        output += m_bias;
    }
    return output;
}

// Matrix Linear::backward(Matrix &gradient)
// {
//     return ;
// }
