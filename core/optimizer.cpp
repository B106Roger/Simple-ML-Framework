#include "optimizer.h"

void SGD::apply_gradient(Network &network, std::vector<std::vector<Matrix>> gradients)
{
    std::vector<std::vector<Matrix>> processed_grad = this->process_gradient(gradients);
    network.apply_gradients(processed_grad);
}

std::vector<std::vector<Matrix>> SGD::process_gradient(std::vector<std::vector<Matrix>> gradient)
{
    std::vector<std::vector<Matrix>> new_gradient = gradient;
    for(size_t i = 0; i < new_gradient.size(); i++) 
    {
        for(size_t j = 0; j < new_gradient[i].size(); j++) {
            Matrix &grad=new_gradient[i][j];
            grad=grad*m_learning_rate;
            if (m_previous_grad.size() != 0){
                grad+=m_previous_grad[i][j] * m_momentum;
            }
        }
    }
    m_previous_grad=new_gradient;
    return new_gradient;
}