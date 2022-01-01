#include "network.h"

Network::Network(std::vector<BaseLayer*>  layers) {
    std::cout << "cosntruct network" << std::endl;
    m_layers = layers;
}

Network::Network(std::vector<int>  layers) {
        for(size_t i =0; i < layers.size(); i++) 
            std::cout << layers[i] << std::endl;
    }

Network::~Network(){}

// input_tensor=(batch, feat_dim)
Matrix Network::forward(Matrix input_tensor)
{
    for(size_t i = 0; i < m_layers.size(); i++) {
        input_tensor=(*m_layers[i])(input_tensor);
    }
    return input_tensor;
}

std::vector<pybind11::tuple> Network::backward(Matrix gradient_flow)
{
    std::vector<pybind11::tuple> gradients;
    for(int i = m_layers.size()-1; i >= 0; i--) {
        std::pair<Matrix, pybind11::tuple> return_data = m_layers[i]->backward(gradient_flow);
        gradient_flow= return_data.first;
        gradients.push_back(return_data.second);
    }
    std::reverse(gradients.begin(), gradients.end());
    return gradients;
}

void Network::apply_gradients(std::vector<pybind11::tuple> gradients)
{
    size_t gradient_idx = 0;
    for (size_t i = 0; i < m_layers.size(); i++)
    {
        BaseLayer *layer=m_layers[i];
        if (layer->get_has_trainable_var() && layer->get_trainable()) {
            layer->apply_gradient(gradients[gradient_idx]);
            gradient_idx+=1;
        }
    }
}