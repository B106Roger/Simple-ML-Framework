#include"matrix.h"

class BaseLayer
{
public:
    BaseLayer();
    BaseLayer(bool trainable);
    ~BaseLayer();

    Matrix operator()(Matrix &input_tensor);
    Matrix forward(Matrix &input_tensor);
private:
    Matrix m_input;
    Matrix m_grad;
    bool   m_trainable;
};