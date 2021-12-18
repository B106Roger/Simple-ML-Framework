#include"matrix.h"

class BaseLayer
{
public:
    BaseLayer();
    BaseLayer(bool trainable, bool transpose_input);
    ~BaseLayer();

    Matrix operator()(Matrix &input_tensor);
    
    virtual Matrix forward(Matrix &input_tensor);
    virtual Matrix backward(Matrix &input_tensor);
    // virtual void build();
private:
    Matrix m_input;
    Matrix m_grad;
    bool   m_trainable;
    bool   m_transpose_input;
};