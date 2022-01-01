#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"matrix.h"

#ifndef __BASE_LAYER__
#define __BASE_LAYER__
class BaseLayer
{
public:
    BaseLayer();
    BaseLayer(bool trainable, bool has_trainable_var);
    ~BaseLayer();

    Matrix operator()(Matrix &input_tensor);
    bool get_has_trainable_var() const {return m_has_trainable_var;}
    bool get_trainable() const {return m_trainable;}
    //////////////////////////////////////////////////////////
    /////////////////// Virtual Function /////////////////////
    //////////////////////////////////////////////////////////
    virtual Matrix forward(const Matrix &input_tensor);
    virtual std::pair<Matrix,pybind11::tuple> backward(Matrix &input_tensor);
    virtual void apply_gradient(pybind11::tuple gradients);

protected:
    Matrix m_input;
    bool   m_trainable;
    bool   m_has_trainable_var;
};
#endif
// #ifndef __BASE_LAYER__