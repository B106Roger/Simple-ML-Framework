#include "optimizer.h"

void SGD::apply_gradient(Network &network, std::vector<pybind11::tuple> gradients)
{
    std::vector<pybind11::tuple> processed_grad = this->process_gradient(gradients);
    network.apply_gradients(processed_grad);
}

std::vector<pybind11::tuple> SGD::process_gradient(std::vector<pybind11::tuple> gradient)
{
    std::vector<pybind11::tuple> new_gradient; // =gradient;
    for(size_t i = 0; i < gradient.size(); i++) 
    {
        pybind11::tuple &grads = gradient[i];

        Matrix grad = grads[0].cast<Matrix>();
        grad=grad*m_learning_rate;
        if (m_previous_grad.size() != 0u && m_momentum != 0.0) {
            std::cout << "in if loop\n" ;
            grad+=m_previous_grad[i][0].cast<Matrix>() * m_momentum;
        }
        if (grads.size() == 2){
            Matrix grad2 = grads[1].cast<Matrix>();
            grad2=grad2*m_learning_rate;
            if (m_previous_grad.size() != 0u && m_momentum != 0.0) {
                std::cout << "in if loop\n" ;
                grad2+=m_previous_grad[i][1].cast<Matrix>() * m_momentum;
            }
            new_gradient.push_back(pybind11::make_tuple(grad, grad2));
            continue;
        }
        new_gradient.push_back(pybind11::make_tuple(grad));



        
    }
    m_previous_grad=new_gradient;
    return new_gradient;
}