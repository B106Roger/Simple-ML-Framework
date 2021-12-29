import unittest
import numpy as np
import _matrix as mat
from testcase.test_util import (
    print_matrix,
    init_matrix,
    print_ndarray,
    test_equal,
)


class TestStringMethods(unittest.TestCase):
    def test_layer_construct(self):
        batch_size=1
        layer_shapes=[
            (16,32),
            (32,64),
            # (64,32),
            # (32,64),
            (64,128),
            (128,64),
        ]
        # Initialize dataset, weight
        random_data  =np.random.randint(low=-10, high=10, size=(batch_size, layer_shapes[0][0])).astype(np.float64)
        Mrandom_data  =mat.Matrix(random_data)
        
        for in_feat, out_feat in layer_shapes:
            random_weight=np.random.randint(low=-10, high=10, size=(in_feat, out_feat)).astype(np.float64)
            random_bias  =np.random.randint(low=-10, high=10, size=(1,       out_feat)).astype(np.float64)
            layer1=mat.Linear(in_feat, out_feat, True, True)
            
            # Transform to Matrix type
            Mrandom_weight=mat.Matrix(random_weight)
            Mrandom_bias  =mat.Matrix(random_bias)

            # set layer weight
            layer1.set_weight((Mrandom_weight, Mrandom_bias))

            # calculate result in C++ and numpy
            Mrandom_data=layer1(Mrandom_data)
            random_data=random_data@random_weight+random_bias
            print(Mrandom_data.array.shape, random_data.shape)

            # assert shape equal
            self.assertTrue(random_data.shape == Mrandom_data.array.shape)
            # assert content equal
            self.assertTrue(test_equal(Mrandom_data, random_data))

            # if test_equal(Mrandom_data, random_data):
            #     print(Mrandom_data.array)
            #     print('-'*50)
            #     print(random_data)


if __name__ == '__main__':
    unittest.main()