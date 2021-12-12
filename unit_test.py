import unittest
import numpy as np
import _matrix

def print_matrix(mat):
    r=mat.nrow
    c=mat.ncol
    for i in range(r):
        for j in range(c):
            print(f'{mat[i,j]:10.4f}', end=' ')
        print()

def init_matrix(mat):
    r=mat.nrow
    c=mat.ncol
    for i in range(r):
        for j in range(c):
            mat[i,j]=i*mat.ncol+j

def print_ndarray(ndarray):
    r=ndarray.shape[0]
    c=ndarray.shape[1]
    for i in range(r):
        for j in range(c):
            print(f'{ndarray[i,j]:10.4f}', end=' ')
        print()

def test_equal(mat1, ndarray):
    col=mat1.ncol
    row=mat1.nrow
    for i in range(row):
        for j in range(col):
            if mat1[i,j] != ndarray[i,j]: return False
    return True

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

if __name__ == '__main__':
    unittest.main()