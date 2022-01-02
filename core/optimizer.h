#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "network.h"

#ifndef __OPTIMIZER__
#define __OPTIMIZER__

class SGD
{
public:
    SGD(double learning_rate, double momentum): 
        m_learning_rate(learning_rate), m_momentum(momentum) {}
    void apply_gradient(Network &network, std::vector<std::vector<Matrix>> gradients);

private:
    std::vector<std::vector<Matrix>> process_gradient(std::vector<std::vector<Matrix>> gradients); 
    std::vector<std::vector<Matrix>> m_previous_grad;
    double m_learning_rate;
    double m_momentum;
};

#endif