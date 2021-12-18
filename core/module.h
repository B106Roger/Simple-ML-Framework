#include<vector>
#include "base_layer.h"

class Module
{
public:
    Module(std::vector<BaseLayer> layers);
    Module~();
    Matrix operator()(Matrix &input_tensor);

private: 
    Matrix forward(Matrix &input_tensor);


    std::vector<BaseLayer> m_layers;
}