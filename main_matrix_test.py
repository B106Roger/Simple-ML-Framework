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
        row=10
        col=10
        # Matrix Library Result
        a=_matrix.Matrix(row,col)
        init_matrix(a)
        b=_matrix.multiply_naive(a, a)
        # Numpy Libray Result
        c=np.arange(0, row*col).reshape((row,col))
        d=c@c
        self.assertTrue(test_equal(b,d))

    def test_multiply_tile(self):
        row=10
        col=10
        # Matrix Library Result
        a=_matrix.Matrix(row,col)
        init_matrix(a)
        b=_matrix.multiply_tile(a, a, 4)
        # Numpy Libray Result
        c=np.arange(0, row*col).reshape((row,col))
        d=c@c
        self.assertTrue(test_equal(b,d))

    def test_multiply_mkl(self):
        row=10
        col=10
        # Matrix Library Result
        a=_matrix.Matrix(row,col)
        init_matrix(a)
        b=_matrix.multiply_mkl(a, a)
        # Numpy Libray Result
        c=np.arange(0, row*col).reshape((row,col))
        d=c@c
        self.assertTrue(test_equal(b,d))

    def test_equal(self):
        row=10
        col=10
        a=_matrix.Matrix(row,col)
        b=_matrix.Matrix(row,col)
        init_matrix(a)
        init_matrix(b)
        self.assertTrue(a==b)
        self.assertFalse(id(a) == id(b))
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

    def test_operator_matrix_double(self):
        print()
        row=10
        col=20
        num=99

        for op in ['__add__', '__sub__','__mul__','__truediv__']:
            # Matrix Library Result
            a=_matrix.Matrix(row,col)
            init_matrix(a)
            b=getattr(a, op)(num)
            # Numpy Libray Result
            c=np.arange(0, row*col).reshape((row,col))
            d=getattr(c, op)(num)
            self.assertTrue(test_equal(b,d))
            print(f'{op:12s} PASS')

    def test_operator_matrix_matrix(self):
        print()
        row=10
        col=20
        for op in ['__add__', '__sub__','__mul__','__truediv__']:
            # Matrix Library Result
            a=_matrix.Matrix(row,col)
            init_matrix(a)
            a = a + 1
            b=getattr(a, op)(a)
            # Numpy Libray Result
            c=np.arange(0, row*col).reshape((row,col)) + 1
            d=getattr(c, op)(c)
            self.assertTrue(test_equal(b,d))
            print(f'{op:12s} PASS')

    def test_operator_double_matrix(self):
        print()
        row=10
        col=20
        num=99
        a=_matrix.Matrix(row,col)
        init_matrix(a)
        a = a + 1
        c=np.arange(1, row*col+1).reshape((row,col))

        # Matrix Library Result
        b=num + a
        # Numpy Libray Result
        d=num + c
        self.assertTrue(test_equal(b,d))
        print(f'{"__add__":12s} PASS')

        # Matrix Library Result
        b=num - a
        # Numpy Libray Result
        d=num - c
        self.assertTrue(test_equal(b,d))
        print(f'{"__sub__":12s} PASS')

        # Matrix Library Result
        b=num * a
        # Numpy Libray Result
        d=num * c
        self.assertTrue(test_equal(b,d))
        print(f'{"__mul__":12s} PASS')

        # Matrix Library Result
        b=num / a
        # Numpy Libray Result
        d=num / c
        self.assertTrue(test_equal(b,d))
        print(f'{"__truediv__":12s} PASS')

    def test_operator_assign_matrix_double(self):
        print()
        row=10
        col=20
        num=99
        a=_matrix.Matrix(row,col)
        init_matrix(a)
        c=np.arange(0, row*col).reshape((row,col)).astype(np.float64)

        # Matrix Library Result
        a+=num
        # Numpy Libray Result
        c+=num
        self.assertTrue(test_equal(a,c))
        print(f'{"__IADD__":12s} PASS, {c.sum():10.2f}')

        # Matrix Library Result
        a-=num
        # Numpy Libray Result
        c-=num
        self.assertTrue(test_equal(a,c))
        print(f'{"__ISUB__":12s} PASS, {c.sum():10.2f}')


        # Matrix Library Result
        a*=num
        # Numpy Libray Result
        c*=num
        self.assertTrue(test_equal(a,c))
        print(f'{"__IMUL__":12s} PASS, {c.sum():10.2f}')

        # Matrix Library Result
        a /= num
        # Numpy Libray Result
        c /= num
        self.assertTrue(test_equal(a,c))
        print(f'{"__IDIV__":12s} PASS, {c.sum():10.2f}')

    def test_operator_assign_matrix_matrix(self):
        print()
        row=10
        col=20
        a=_matrix.Matrix(row,col)
        init_matrix(a)
        a = a + 1

        b=_matrix.Matrix(col,row)
        init_matrix(b)
        b = b + 1
        b=b.T()

        c=np.arange(1, row*col+1).reshape((row,col)).astype(np.float64)
        d=np.arange(1, row*col+1).reshape((col,row)).astype(np.float64).T

        # Matrix Library Result
        a+=b
        # Numpy Libray Result
        c+=d
        self.assertTrue(test_equal(a,c))
        print(f'{"__IADD__":12s} PASS, {c.sum():10.2f}')

        # Matrix Library Result
        a-=b
        # Numpy Libray Result
        c-=d
        self.assertTrue(test_equal(a,c))
        print(f'{"__ISUB__":12s} PASS, {c.sum():10.2f}')


        # Matrix Library Result
        a*=b
        # Numpy Libray Result
        c*=d
        self.assertTrue(test_equal(a,c))
        print(f'{"__IMUL__":12s} PASS, {c.sum():10.2f}')

        # Matrix Library Result
        a /= b
        # Numpy Libray Result
        c /= d
        self.assertTrue(test_equal(a,c))
        print(f'{"__IDIV__":12s} PASS, {c.sum():10.2f}')

if __name__ == '__main__':
    unittest.main()