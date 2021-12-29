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

    virtual Matrix forward(const Matrix &input_tensor);
    virtual Matrix backward(Matrix &input_tensor);

private:
    Matrix m_input;
    Matrix m_grad;
    bool   m_trainable;
    bool   m_transpose_input;
};
#endif
// #ifndef __BASE_LAYER__