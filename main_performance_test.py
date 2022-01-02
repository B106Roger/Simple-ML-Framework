import unittest
import numpy as np
from numpy.core import numeric
import _matrix
import timeit
from testcase.test_util import (
    print_matrix,
    init_matrix,
    print_ndarray,
    test_equal,
)


class TestStringMethods(unittest.TestCase):
    loop=1000
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

    def test_operator_matrix_double(self):
        print()
        row=10
        col=20
        num=np.float64(99)

        a=_matrix.Matrix(row,col)
        init_matrix(a)

        c=np.arange(0, row*col).reshape((row,col)).astype(np.float64)

        for op in ['__mul__']:
            # Matrix Library Result
            print(timeit.timeit(lambda:getattr(a, op)(num), number=self.loop))
            # Numpy Libray Result
            print(timeit.timeit(lambda:getattr(c, op)(num), number=self.loop))
            print(f'{op:12s} PASS')

    def test_operator_matrix_matrix(self):
        print()
        row=32
        col=32
        num = 99

        int32=np.int32(num)
        fp64=np.float64(num)

        a=_matrix.Matrix(row,col)
        init_matrix(a)
        a = a + 1
        b = _matrix.Matrix(a)

        c=np.arange(0, row*col).reshape((row,col)).astype(np.float64) + 1
        d=np.copy(c)

        a_time_list=[]
        b_time_list=[]
        c_time_list=[]
        d_time_list=[]
        for iidx in range(40):
            op='__mul__'
            # Matrix Library Result
            a_time=timeit.timeit(lambda:getattr(a, op)(fp64), number=self.loop)
            b_time=timeit.timeit(lambda:getattr(a, op)(int32), number=self.loop)

            # print(f"{'Matrix = Matrix + Matrix':25s}, {timeit.timeit(lambda:getattr(a, op)(a ), number=self.loop):8.6f}")
            # print(f"{'Matrix = Matrix + Matrix':25s}, {timeit.timeit(lambda:getattr(a, op)(b ), number=self.loop):8.6f}")
            # print(f"{'Matrix = Matrix + double':25s}, {a_time:8.6f}")
            # print(f"{'Matrix = Matrix + int   ':25s}, {b_time:8.6f}")
            # print(f"diff: {b_time-a_time:8.6f}")
            a_time_list.append(a_time)
            b_time_list.append(b_time)


            # print()
            # Numpy Libray Result
            c_time=timeit.timeit(lambda:getattr(c, op)(fp64), number=self.loop)
            d_time=timeit.timeit(lambda:getattr(c, op)(int32), number=self.loop)

            # print(f"{'NP_FP64 + NP_FP64':25s}, {timeit.timeit(lambda:getattr(c, op)(c ), number=self.loop):8.6f}")
            # print(f"{'NP_FP64 + NP_FP64':25s}, {timeit.timeit(lambda:getattr(c, op)(d ), number=self.loop):8.6f}")
            # print(f"{'NP_FP64 + double ':25s}, {c_time:8.6f}")
            # print(f"{'NP_FP64 + int    ':25s}, {d_time:8.6f}")
            c_time_list.append(c_time)
            d_time_list.append(d_time)
            
            print(f'{op:12s} PASS')

        print('Matrix = Matrix + double',np.mean(a_time_list))
        print('Matrix = Matrix + int   ',np.mean(b_time_list))
        print('NP_FP64 + double ',np.mean(c_time_list))
        print('NP_FP64 + int    ',np.mean(d_time_list))


if __name__ == '__main__':
    unittest.main()