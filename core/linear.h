#include <pybind11/pybind11.h>
#include"base_layer.h"
#include"matrix.h"

#ifndef __LINEAR__
#define __LINEAR__
class Linear: public BaseLayer
{
public:
    Linear(int in, int out, bool use_bias=false, bool trainable=true);
    ~Linear();

    Matrix forward(const Matrix &input_tensor);
    // Matrix backward(Matrix &gradient);

    void set_weight(pybind11::tuple py_tuple);
    pybind11::tuple get_weight() {return pybind11::make_tuple(m_weight, m_bias);}
    Matrix& weight() {return m_weight;}
    Matrix& bias() {return m_bias;}
private:

    size_t m_in_feat;
    size_t m_out_feat;
    bool m_use_bias;
    Matrix m_weight;
    Matrix m_bias;
};
#endif
// end #ifndef __LINEAR__
