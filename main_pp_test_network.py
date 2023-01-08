import unittest
import numpy as np
import _matrix as mat
import timeit
import time
from testcase.test_util import (
    init_matrix,
    rand_init_matrix,
    test_equal,
    print_matrix,
    print_ndarray
)


class TestStringMethods(unittest.TestCase):
    loop=10
    row=1024
    col=1024
    batch=128
    def test_multiply_tile1(self):        
        data=mat.Matrix(self.batch, 784)
        
        layers=[
            mat.Linear(784, 128, True, True),
            mat.Sigmoid(),
            mat.Linear(128, 10, True, True),
            mat.Sigmoid(),
        ]
        network=mat.Network(layers)
        
        ##########################################
        # MKL Library Result (Upper Bound)
        ##########################################
        mat.set_matrix_mode(2)
        mlutiply_mkl_time = timeit.timeit(lambda:network(data), number=self.loop, timer=time.process_time) / self.loop
        print(f'mlutiply_mkl_time   : {mlutiply_mkl_time:8.6f}')
        
        ##########################################
        # Naive Method Result (Lower Bound)
        ##########################################
        NAIVE_LOOPS=1
        mat.set_matrix_mode(1)
        multiply_naive_time = timeit.timeit(lambda:network(data), number=NAIVE_LOOPS, timer=time.process_time) / NAIVE_LOOPS
        print(f'multiply_naive_time : {multiply_naive_time:8.6f}')
        
        ##########################################
        # Tiled Matrix Method Result
        ##########################################
        # for tile in [4,8,16,32,64,128]:
        mat.set_matrix_mode(3)
        for tile in [16]:
            multiply_tile_time = timeit.timeit(lambda:network(data), number=self.loop, timer=time.process_time) / self.loop
            print(f'multiply_tile_mod_{tile:03d}   : {multiply_tile_time:8.6f}')
        
        ##########################################
        # Your Accelerate Method Result
        ##########################################
        mat.set_matrix_mode(4)
        multiply_tile_pthread_time = timeit.timeit(lambda:network(data), number=self.loop, timer=time.process_time) / self.loop
        print(f'multiply_tile_016_thread_{8:03d}_pthread : {multiply_tile_pthread_time:8.6f}')
        
        mat.set_matrix_mode(5)
        multiply_tile_SSE_time = timeit.timeit(lambda:network(data), number=self.loop, timer=time.process_time) / self.loop
        print(f'multiply_tile_SSE_{tile:03d}_pthread   : {multiply_tile_SSE_time:8.6f}')
    
        mat.set_matrix_mode(6)
        multiply_tile_AVX_time = timeit.timeit(lambda:network(data), number=self.loop, timer=time.process_time) / self.loop
        print(f'multiply_tile_AVX_{tile:03d}_pthread   : {multiply_tile_AVX_time:8.6f}')
    

    # def test_multiply_tile_backup(self):
    #     print()
    #     # Numpy Libray Result
    #     c=np.arange(0, self.row*self.col).reshape((self.row,self.col)).astype(np.float64)
    #     d=np.arange(0, self.row*self.col).reshape((self.col,self.row)).astype(np.float64)
    #     ans = c @ d
    #     mlutiply_numpy_time = timeit.timeit(lambda: c @ d, number=self.loop, timer=time.process_time)
    #     print(f'mlutiply_numpy_time : {mlutiply_numpy_time:8.6f}')
        
    #     a=_matrix.Matrix(self.row,self.col)
    #     b=_matrix.Matrix(self.col,self.row)
    #     init_matrix(a)
    #     init_matrix(b)
        
    #     ##########################################
    #     # MKL Library Result (Upper Bound)
    #     ##########################################
    #     mlutiply_mkl_time = timeit.timeit(lambda:_matrix.multiply_mkl(a, b), number=self.loop, timer=time.process_time)
    #     print(f'mlutiply_mkl_time   : {mlutiply_mkl_time:8.6f}')
        
    #     ##########################################
    #     # Naive Method Result (Lower Bound)
    #     ##########################################
    #     multiply_naive_time = timeit.timeit(lambda:_matrix.multiply_naive(a, b), number=self.loop, timer=time.process_time)
    #     print(f'multiply_naive_time : {multiply_naive_time:8.6f}')
        
    #     ##########################################
    #     # Tiled Matrix Method Result
    #     ##########################################
    #     for tile in [2,4,8,16,32,64,128]:
    #         t=_matrix.multiply_tile_modify(a, b, tile)
    #         self.assertTrue(test_equal(t,ans))
    #         multiply_tile_time = timeit.timeit(lambda:_matrix.multiply_tile_modify(a, b, tile), number=self.loop, timer=time.process_time)
    #         print(f'multiply_tile_mod_{tile:03d}   : {multiply_tile_time:8.6f}')
        
    #     ##########################################
    #     # Your Accelerate Method Result
    #     ##########################################
    #     for tile in [2,4,8,16,32,64,128]:
    #         t=_matrix.multiply_tile_modify_thread(a, b, tile)
    #         # self.assertTrue(test_equal(t,ans))
    #         multiply_tile_thread_time = timeit.timeit(lambda:_matrix.multiply_tile_modify_thread(a, b, tile), number=self.loop, timer=time.process_time)
    #         print(f'multiply_tile_mod_{tile:03d}_thread   : {multiply_tile_thread_time:8.6f}')
        
    #     for tile in [2,4,8,16,32,64,128]:
    #         t=_matrix.multiply_tile_modify_pthread(a, b, tile)
    #         # self.assertTrue(test_equal(t,ans))
    #         multiply_tile_pthread_time = timeit.timeit(lambda:_matrix.multiply_tile_modify_pthread(a, b, tile), number=self.loop, timer=time.process_time)
    #         print(f'multiply_tile_mod_{tile:03d}_pthread   : {multiply_tile_pthread_time:8.6f}')
        
    # def test_mat_multiply(self):
    #     print()
    #     # Numpy Libray Result
    #     c=np.arange(0, self.row*self.col).reshape((self.row,self.col)).astype(np.float64)
    #     d=np.arange(0, self.row*self.col).reshape((self.col,self.row)).astype(np.float64)
    #     ans = c @ d
    #     mlutiply_numpy_time = timeit.timeit(lambda: c @ d, number=self.loop, timer=time.process_time)
    #     print(f'mlutiply_numpy_time : {mlutiply_numpy_time:8.6f}')
        
    #     a=_matrix.Matrix(self.row,self.col)
    #     b=_matrix.Matrix(self.col,self.row)
    #     init_matrix(a)
    #     init_matrix(b)
        
    #     # Naive Method Result (Lower Bound)
    #     _matrix.set_matrix_mode(1)
    #     multiply_naive_time = timeit.timeit(lambda:_matrix.mat_multiply(a, b), number=self.loop, timer=time.process_time)
    #     print(f'multiply_naive_time : {multiply_naive_time:8.6f}')
        
    #     # MKL Library Result (Upper Bound)
    #     _matrix.set_matrix_mode(2)
    #     mlutiply_mkl_time = timeit.timeit(lambda:_matrix.mat_multiply(a, b), number=self.loop, timer=time.process_time)
    #     print(f'mlutiply_mkl_time   : {mlutiply_mkl_time:8.6f}')
        
    #     # Tiled Matrix Method Result
    #     _matrix.set_matrix_mode(3)
    #     t=_matrix.mat_multiply(a, b)
    #     # self.assertTrue(test_equal(t,ans))
    #     multiply_tile_time = timeit.timeit(lambda:_matrix.mat_multiply(a, b), number=self.loop, timer=time.process_time)
    #     print(f'multiply_tile_mod_{32}   : {multiply_tile_time:8.6f}')
        

        

if __name__ == '__main__':
    unittest.main()