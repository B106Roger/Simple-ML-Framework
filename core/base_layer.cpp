#include"base_layer.h"


BaseLayer::BaseLayer(){}
BaseLayer::BaseLayer(bool trainable, bool transpose_input)
    : m_trainable(trainable), m_transpose_input(transpose_input){}
BaseLayer::~BaseLayer(){}

// input_tensor=(batch, feat_dim)
Matrix BaseLayer::operator()(Matrix &input_tensor)
{
    this->m_input = input_tensor;
    return this->forward(input_tensor);
}

Matrix BaseLayer::forward(const Matrix &input_tensor)
{
    std::cout << "calling BaseLayer forward" << std::endl;
    this->m_input = input_tensor;
    return this->forward(input_tensor);
}

std::pair<Matrix,pybind11::tuple> BaseLayer::backward(Matrix &gradient)
{
    return std::pair<Matrix,pybind11::tuple>(
        gradient, 
        pybind11::make_tuple(gradient)
    );
}
