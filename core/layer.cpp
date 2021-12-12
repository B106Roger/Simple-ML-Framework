#include"layer.h"


BaseLayer::BaseLayer(){}
BaseLayer::BaseLayer(bool trainable): m_trainable(trainable){}
BaseLayer::~BaseLayer(){}

Matrix BaseLayer::operator()(Matrix &input_tensor)
{
    this->m_input = input_tensor;
    Matrix output_tensor = this->forward(input_tensor);
    return output_tensor;
}
