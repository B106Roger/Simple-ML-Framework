#include"base_layer.h"
#include"matrix.h"

class Linear: public BaseLayer
{
public:
    Linear(int in, int out, bool use_bias=false, bool trainable=true);
    ~Linear();

    Matrix forward(Matrix &input_tensor);
    // Matrix backward(Matrix &gradient);
    // void build();
private:

    int m_in_feat;
    int m_out_feat;
    bool m_use_bias;
    Matrix m_weight;
    Matrix m_bias;
}