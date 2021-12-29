#include <pybind11/stl.h>
#include "base_layer.h"

#ifndef __NETWORK__
#define __NETWORK__
class Network
{
public:
    Network(std::vector<BaseLayer*> layers);
    Network(std::vector<int> layers);

    ~Network();
    Matrix forward(Matrix input_tensor);
    // Matrix backwrad(Matrix gradient);
    std::vector<BaseLayer*>& get_layers() {return m_layers; }

private: 
    // Matrix forward(Matrix input_tensor);
    std::vector<BaseLayer*> m_layers;

};
#endif