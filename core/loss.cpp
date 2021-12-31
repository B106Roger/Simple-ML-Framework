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