#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "base_layer.h"

#ifndef __NETWORK__
#define __NETWORK__
class Network
{
public:
    Network(std::vector<BaseLayer*> layers);
    
    ~Network();
    Matrix forward(Matrix input_tensor);
    std::vector<std::vector<Matrix>> backward(Matrix gradient);
    std::vector<BaseLayer*>& get_layers() {return m_layers; }
    void apply_gradients(std::vector<std::vector<Matrix>> gradients);

private:
    // Matrix forward(Matrix input_tensor);
    std::vector<BaseLayer*> m_layers;

};
#endif