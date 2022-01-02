#include"matrix.h"
#include"loss.h"

////////////////////////////////////////////////////////////
///////////////////     BaseLoss       /////////////////////
////////////////////////////////////////////////////////////
Matrix BaseLoss::operator()(const Matrix &prediction, const Matrix &ground_truth)
{
    this->m_input=prediction;
    return this->forward(prediction, ground_truth);
}

Matrix BaseLoss::forward(const Matrix &prediction, const Matrix &ground_truth)
{
    return ground_truth;
}

Matrix BaseLoss::backward()
{
    return m_gradient;
}


////////////////////////////////////////////////////////////
///////////////////         MSE        /////////////////////
////////////////////////////////////////////////////////////
Matrix MSE::forward(const Matrix &prediction, const Matrix &ground_truth)
{
    Matrix result=(prediction-ground_truth).power(2.0);
    m_gradient = (prediction - ground_truth) * 2.0;
    return result;
}
Matrix MSE::backward()
{
    return m_gradient;
}

////////////////////////////////////////////////////////////
////////////    CategoricalCrossentropy   //////////////////
////////////////////////////////////////////////////////////
Matrix CategoricalCrossentropy::forward(const Matrix &prediction, const Matrix &ground_truth)
{
    Matrix mat_exp = prediction.exp();
    Matrix mat_exp_sum(prediction.nrow(), 1);
    for(size_t i = 0; i < prediction.nrow(); i++) {
        for(size_t j = 0; j < prediction.ncol(); j++) {
            mat_exp_sum(i,0) += mat_exp(i,j); 
        }
    }
    // broadcasting
    Matrix normalize = mat_exp / mat_exp_sum;
    m_gradient = normalize - ground_truth;
    return normalize.log() * ground_truth * -1.0;
}
Matrix CategoricalCrossentropy::backward()
{
    return m_gradient;
}