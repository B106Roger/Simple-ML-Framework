#include "base_layer.h"


class Sigmoid: public BaseLayer
{
public:
    Sigmoid(): BaseLayer(false, false) {}
    
    //////////////////////////////////////////////////////////
    /////////////////// Virtual Function /////////////////////
    //////////////////////////////////////////////////////////
    Matrix forward(const Matrix &input_tensor)
    {
        return input_tensor.sigmoid();
    }
    std::pair<Matrix, std::vector<Matrix>> backward(Matrix &gradient)
    {
        Matrix gradient_flow = m_input.sigmoid().power(2.0) * (m_input*-1.0).exp() * gradient;
        return std::pair<Matrix, std::vector<Matrix>>(gradient_flow, {});
    }
    pybind11::dict get_config() const {
        return pybind11::dict(
            "has_var"_a=m_has_trainable_var
        );
    }
};

class ReLU: public BaseLayer
{
public:
    ReLU(): BaseLayer(false, false) {}
    
    //////////////////////////////////////////////////////////
    /////////////////// Virtual Function /////////////////////
    //////////////////////////////////////////////////////////
    Matrix forward(const Matrix &input_tensor)
    {
        return input_tensor.relu();
    }
    std::pair<Matrix, std::vector<Matrix>> backward(Matrix &gradient)
    {
        Matrix gradient_flow = gradient;
        for(size_t i = 0; i < gradient_flow.nrow(); i++) {
            for(size_t j = 0; j < gradient_flow.ncol(); j++) {
                if (m_input(i,j) < 0.0)
                    gradient_flow(i,j) = 0.0;
            }
        }
        return std::pair<Matrix, std::vector<Matrix>>(gradient_flow, {});
    }
    pybind11::dict get_config() const {
        return pybind11::dict(
            "has_var"_a=m_has_trainable_var
        );
    }
};