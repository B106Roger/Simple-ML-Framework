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
    // std::cout << "calling network forward" << std::endl;
    // input_tensor.print_shape();
    // input_tensor=(batch, feat_dim)
    for(size_t i = 0; i < m_layers.size(); i++) {
        input_tensor=(*m_layers[i])(input_tensor);
        // input_tensor.print_shape();
    }
    return input_tensor;
}

std::vector<pybind11::tuple> Network::backward(Matrix gradient_flow)
{
    std::vector<pybind11::tuple> gradients;
    for(int i = m_layers.size()-1; i >= 0; i--) {
        // std::cout<<"backward layer: " << i << " start" << std::endl;
        std::pair<Matrix, pybind11::tuple> return_data = m_layers[i]->backward(gradient_flow);
        gradient_flow= return_data.first;
        gradients.push_back(return_data.second);
        // std::cout<<"backward layer: " << i << " end" << std::endl;
    }
    std::cout<<"network backward end" << std::endl;
    // for(BaseLayer *layer: m_layers)
    // {
    //     std::pair<Matrix, pybind11::tuple> return_data = layer->backward(gradient_flow);
    //     gradient_flow= return_data.first;
    //     gradients.push_back(return_data.second);
    // }
    std::reverse(gradients.begin(), gradients.end());
    return gradients;
}