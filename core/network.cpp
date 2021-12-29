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
    std::cout << "calling network forward" << std::endl;
    input_tensor.print_shape();
    // input_tensor=(batch, feat_dim)
    for(size_t i = 0; i < m_layers.size(); i++) {
        std::cout << "for loop " << i << std::endl;
        input_tensor=m_layers[i]->forward(input_tensor);
        input_tensor.print_shape();
    }
    // for(BaseLayer* layer: m_layers)
    // {
    //     input_tensor=layer->forward(input_tensor);
    //     input_tensor.print_shape();
    // }
    return input_tensor;
}