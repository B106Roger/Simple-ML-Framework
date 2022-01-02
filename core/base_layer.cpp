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

std::pair<Matrix,std::vector<Matrix>> BaseLayer::backward(Matrix &gradient)
{
    return std::pair<Matrix,std::vector<Matrix>>(
        gradient, 
        {gradient}
    );
}

void BaseLayer::apply_gradient(std::vector<Matrix> gradients)
{

}

void BaseLayer::set_weight(std::vector<Matrix> weight_list)
{

}

std::vector<Matrix> BaseLayer::get_weight()
{
    return {};
}

pybind11::dict BaseLayer::get_config() const
{
    return pybind11::dict();
}