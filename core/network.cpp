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

std::vector<std::vector<Matrix>> Network::backward(Matrix gradient_flow)
{
    std::vector<std::vector<Matrix>> gradients;
    for(int i = m_layers.size()-1; i >= 0; i--) {
        std::pair<Matrix, std::vector<Matrix>> return_data = m_layers[i]->backward(gradient_flow);
        gradient_flow= return_data.first;
        if (!m_layers[i]->get_has_trainable_var() && return_data.second.size() != 0)
            throw std::runtime_error("no variable layaer should not have variable gradient\n");
        gradients.push_back(return_data.second);
    }
    std::reverse(gradients.begin(), gradients.end());
    return gradients;
}

void Network::apply_gradients(std::vector<std::vector<Matrix>> gradients)
{
    for (size_t i = 0; i < m_layers.size(); i++)
    {
        BaseLayer *layer=m_layers[i];
        if (layer->get_trainable()) {
            layer->apply_gradient(gradients[i]);
        }
    }
}