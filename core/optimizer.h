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
    void apply_gradient(Network &network, std::vector<pybind11::tuple> gradients);

private:
    std::vector<pybind11::tuple> process_gradient(std::vector<pybind11::tuple> gradients); 
    std::vector<pybind11::tuple> m_previous_grad;
    double m_learning_rate;
    double m_momentum;
};

#endif