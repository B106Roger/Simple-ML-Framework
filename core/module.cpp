#include "module.h"

Module::Module(std::vector<BaseLayer>  layers)
    :m_layers(layers) {}

Module::~Module(){}

// input_tensor=(batch, feat_dim)
Matrix Module::operator()(Matrix input_tensor)
{
    // input_tensor=(feat_dim, batch)
    input_tensor=input_tensor.T();
    for(BaseLayer& layer: m_layers)
    {
        input_tensor=layer(input_tensor);
    }
    return input_tensor;
}