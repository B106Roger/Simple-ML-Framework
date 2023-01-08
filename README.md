# Simple ML Framework
## Installation
Use python3.8 to prevent some package compatibility issue
```
https://github.com/B106Roger/Simple-ML-Framework.git
cd Simple-ML-Framework
pip3 install -r requirements.txt
```
----
## Build Project
```
make
```
*note that if you aren't using conda to manage environment, you should change the below include path in make by yourself !!!*
```=bash
###################################
# MKL Library Header File Path
###################################
MKL_INCLUDE_DIRS =  $(CONDA_PREFIX)/include

###################################
# MKL Library LIB Path
###################################
# MKL_LIBS = -lmkl_def -lmkl_avx2 -lmkl_core -lmkl_intel_lp64 -lmkl_sequential
MKL_LIB_DIR = $(CONDA_PREFIX)/lib
MKL_LIBS = ${MKL_LIB_DIR}/libmkl_def.so.2 \
		   ${MKL_LIB_DIR}/libmkl_avx2.so.2 \
		   ${MKL_LIB_DIR}/libmkl_core.so.2 \
		   ${MKL_LIB_DIR}/libmkl_intel_lp64.so.2 \
		   ${MKL_LIB_DIR}/libmkl_sequential.so.2
```
----
## Create Your Own Matrix Multiplication Method

### Step1 Create Matrix Multiplication Function

1. put your implementation in matrix.cpp

core/matrix.cpp
```=c++
// Accelerate Part
//////////////////////////////////////////////////////////////////////////////////////////
// Create Your Own Matrix Multiplication Below
// Note that all the Matrix Multiplication should have signature like
// Matrix multiply_YOUR_FUNC_NAME(const Matrix &mat1, const Matrix &mat2, ...other-argument) 
//////////////////////////////////////////////////////////////////////////////////////////
Matrix multiply_YOUR_FUNC_NAME(const Matrix &mat1, const Matrix &mat2, ...other-argument)
{
	size_t row=mat1.nrow();
    size_t col=mat2.ncol();
    size_t content=mat1.ncol();
    Matrix result(row, col);
	......
	......
	......
	return result;
}
```

2. don't forget your declaration in matrix.h

core/matrix.h
```=c++
// Accelerate Part
//////////////////////////////////////////////////////////////////////////////////////////
// Create Your Own Matrix Multiplication Below
// Note that all the Matrix Multiplication should have signature like
// Matrix multiply_YOUR_FUNC_NAME(const Matrix &mat1, const Matrix &mat2, ...other-argument) 
//////////////////////////////////////////////////////////////////////////////////////////
Matrix multiply_YOUR_FUNC_NAME(const Matrix &mat1, const Matrix &mat2, ...other-argument);
```

### Step2 Expose Your Matrix Multiplication Function via pybind11
1. register your matrix multiplication function\

core/pybindwrapper.cpp
```=c++
// Accelerate Part
//////////////////////////////////////////////////////////////////////////////////////////
// Register Your Own Matrix Multiplication Below
//////////////////////////////////////////////////////////////////////////////////////////
m.def("multiply_YOUR_FUNC_NAME", &multiply_YOUR_FUNC_NAME);
```

2. Don't Forget to compile the project again
```=bash
make clean
make
```
### Step3 Write Test Script and Testing in Python
1. main_pp_performance_test.py
```=python
##########################################
# Your Accelerate Method Result
##########################################
t=_matrix.multiply_YOUR_FUNC_NAME(a, b)
self.assertTrue(test_equal(t,ans))
elipse_time = timeit.timeit(lambda:_matrix.multiply_YOUR_FUNC_NAME(a, b), number=self.loop, timer=time.process_time)
print(f'multiply_YOUR_FUNC_NAME   : {elipse_time:8.6f}')
```

2. run command in root
```=bash
python -m unittest main_pp_performance_test.py -v
```

### Step4 Run Gradient Descent
1. Install dataset (train_data.npz and test_data.npz) and put them under testcase folder
https://drive.google.com/drive/folders/10pa9nPWKx6DtnjBlM5zUsxY31EodeZC7?usp=share_link

2. run command in root
```=bash
python ./main.py
```

### Step 5 Run Gradient Descent Using Your Accelerated Matrix Multiplication
1. Register Your Function in mat_multiply
by default, forward and backpropagation use _matrx.mat_multiply to do calculation.\
please add new option for the function

core/matrix.cpp
```
// Tested Part
Matrix mat_multiply(const Matrix &mat1, const Matrix &mat2)
{
    switch (Matrix::multiplication_mode)
    {
        case 1:
            return multiply_naive(mat1, mat2);
        case 2:
            return multiply_mkl(mat1, mat2);
        case 3: 
            return multiply_tile_modify(mat1, mat2, 32);
		// Add New Option Below !!
		case 4:
			return multiply_YOUR_FUNC_NAME(ma1, mat2, ...other-argument);
    }
    return multiply_naive(mat1, mat2);
}
``` 

2. Set Your Matrix Mode Before Training Model

main.py
```=python
import numpy as np
import _matrix as mat
import matplotlib.pyplot as plt
import cv2
from testcase.test_util import (
    AvgCounter
)
#############################################
# Set Your Matrix Multiplication Mode 
# by default using MKL multiplication
#############################################
# _matrix.set_matrix_mode(YOUR_DESICRED_MODE)
_matrix.set_matrix_mode(4)
```

----
## Testing Code
```
make test
```
----
## Final Project Presentation of nsdhw 21au
final presentation: https://docs.google.com/presentation/d/1jQUTzosQuHBJyuUYS2R96sUD37GK-WjS/edit?usp=sharing&ouid=105896444551749782879&rtpof=true&sd=true
## Final Project Presentation of PP 22au
final presentation: https://docs.google.com/presentation/d/1sYtRUcwCqLc0KEsUcMUCjN1NqgxUn9j9/edit?usp=sharing&ouid=105896444551749782879&rtpof=true&sd=true