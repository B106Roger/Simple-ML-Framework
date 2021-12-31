#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"matrix.h"

#ifndef __BASE_LAYER__
#define __BASE_LAYER__
class BaseLayer
{
public:
    BaseLayer();
    BaseLayer(bool trainable, bool transpose_input);
    ~BaseLayer();

    Matrix operator()(Matrix &input_tensor);
    //////////////////////////////////////////////////////////
    /////////////////// Virtual Function /////////////////////
    //////////////////////////////////////////////////////////
    virtual Matrix forward(const Matrix &input_tensor);
    virtual std::pair<Matrix,pybind11::tuple> backward(Matrix &input_tensor);

protected:
    Matrix m_input;
    bool   m_trainable;
    bool   m_transpose_input;
};
#endif
// #ifndef __BASE_LAYER__