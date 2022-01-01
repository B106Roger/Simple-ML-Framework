#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include"base_layer.h"
#include"matrix.h"

#ifndef __LINEAR__
#define __LINEAR__
class Linear: public BaseLayer
{
public:
    using BaseLayer::BaseLayer;
    Linear(int in, int out, bool use_bias=false, bool trainable=true);
    ~Linear();

    void set_weight(pybind11::tuple py_tuple);
    pybind11::tuple get_weight() {return pybind11::make_tuple(m_weight, m_bias);}
    Matrix& weight() {return m_weight;}
    Matrix& bias() {return m_bias;}

    //////////////////////////////////////////////////////////
    /////////////////// Virtual Function /////////////////////
    //////////////////////////////////////////////////////////
    Matrix forward(const Matrix &input_tensor);
    std::pair<Matrix,pybind11::tuple> backward(Matrix &gradient);
    void apply_gradient(pybind11::tuple gradients);

private:
    // configuration parameter
    size_t m_in_feat;
    size_t m_out_feat;
    bool m_use_bias;
    // runtime parameter
    Matrix m_weight;
    Matrix m_bias;
    Matrix m_weight_gradient;
    Matrix m_bias_gradient;

};
#endif
// end #ifndef __LINEAR__
