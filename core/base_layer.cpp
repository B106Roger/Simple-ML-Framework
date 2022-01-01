#include"base_layer.h"


BaseLayer::BaseLayer(){}
BaseLayer::BaseLayer(bool trainable, bool has_trainable_var)
    : m_trainable(trainable), m_has_trainable_var(has_trainable_var){}
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

void BaseLayer::apply_gradient(pybind11::tuple gradients)
{
    
}