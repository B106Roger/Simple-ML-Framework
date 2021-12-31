#include<iostream>
#include"matrix.h"

#ifndef __LOSS__
#define __LOSS__

class BaseLoss
{
public:
    BaseLoss(){};
    ~BaseLoss(){};
    Matrix operator()(const Matrix &input_tensor, const Matrix &ground_truth);

    virtual Matrix forward(const Matrix &prediction, const Matrix &ground_truth);
    virtual Matrix backward();
protected:
    Matrix m_gradient;
    Matrix m_input;
};


class MSE: public BaseLoss
{
public:
    using BaseLoss::BaseLoss;
    MSE(): BaseLoss() {};
    ~MSE(){};
    Matrix forward(const Matrix &prediction, const Matrix &ground_truth);
    Matrix backward();
};


#endif
