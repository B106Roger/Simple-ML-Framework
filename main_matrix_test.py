import unittest
import numpy as np
import _matrix
from testcase.test_util import (
    print_matrix,
    init_matrix,
    print_ndarray,
    test_equal,
)


class TestStringMethods(unittest.TestCase):
    def test_multiply_naive(self):
        # Matrix Library Result
        a=_matrix.Matrix(10,10)
        init_matrix(a)
        b=_matrix.multiply_naive(a, a)
        # print('Matrix Result From _matrix.multiply_naive')
        # print_matrix(b)
        # Numpy Libray Result
        c=np.arange(0, 100).reshape((10,10))
        d=c@c
        # print('Matrix Result From np.matmul')
        # print_ndarray(d)
        self.assertTrue(test_equal(b,d))

    def test_multiply_tile(self):
        # Matrix Library Result
        a=_matrix.Matrix(10,10)
        init_matrix(a)
        b=_matrix.multiply_tile(a, a, 4)
        # print('Matrix Result From _matrix.multiply_tile')
        # print_matrix(b)
        # Numpy Libray Result
        c=np.arange(0, 100).reshape((10,10))
        d=c@c
        # print('Matrix Result From np.matmul')
        # print_ndarray(d)
        self.assertTrue(test_equal(b,d))

    def test_multiply_mkl(self):
        # Matrix Library Result
        a=_matrix.Matrix(10,10)
        init_matrix(a)
        b=_matrix.multiply_mkl(a, a)
        # print('Matrix Result From _matrix.multiply_mkl')
        # print_matrix(b)
        # Numpy Libray Result
        c=np.arange(0, 100).reshape((10,10))
        d=c@c
        # print('Matrix Result From np.matmul')
        # print_ndarray(d)
        self.assertTrue(test_equal(b,d))

    def test_equal(self):
        a=_matrix.Matrix(10,10)
        b=_matrix.Matrix(10,10)
        init_matrix(a)
        init_matrix(b)
        self.assertTrue(a==b)
        a[1,1]=9999
        self.assertFalse(a==b)

    def test_transpose(self):
        # mat1=np.arange(0,20000).reshape(200,100)
        mat1=np.arange(0,20).reshape(2,10)
        mat1T=mat1.T

        # Test Matrix transpose
        testmat=_matrix.Matrix(mat1)
        testmatT=testmat.T()
        self.assertTrue(test_equal(testmatT, mat1T))
        
        # Test Matrix transpose multiplication
        res=mat1@mat1T
        test_res=_matrix.multiply_mkl(testmat, testmatT)
        self.assertTrue(test_equal(test_res, res))

if __name__ == '__main__':
    unittest.main()