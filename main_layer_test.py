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

    def test_network_construct(self):
        print('*'*70)
        batch=4
        layer_shape=[
            (16,32),
            (32,64),
            (64,128),
            (128,16)
        ]
        a=[
            mat.Linear(*layer_shape[0], True, True),
            mat.Linear(*layer_shape[1], True, True),
            mat.Linear(*layer_shape[2], True, True),
            mat.Linear(*layer_shape[3], True, True),
        ]
        network=mat.Network(a)
        # initialize network
        random_data=np.random.randint(low=-5,high=5,size=(batch, layer_shape[0][0]))
        Mrandom_data=mat.Matrix(random_data)
        result=network(Mrandom_data)
        self.assertTrue(result.array.shape==(batch, layer_shape[-1][-1]))

        for i, (in_feat, out_feat) in enumerate(layer_shape):
            random_weight=np.random.randint(low=-5, high=5, size=(in_feat, out_feat))
            Mrandom_weight=mat.Matrix(random_weight)
            random_bias=np.random.randint(low=-5, high=5, size=(1, out_feat))
            Mrandom_bias=mat.Matrix(random_bias)
            network.layers[i].set_weight((Mrandom_weight, Mrandom_bias))
            random_data=random_data@random_weight+random_bias
        Mrandom_data=network(Mrandom_data)

        self.assertTrue(test_equal(Mrandom_data, random_data))
        self.assertFalse(id(Mrandom_data)==id(random_data))
        print(hex(id(Mrandom_data)), hex(id(random_data)))
        print(Mrandom_data)
        
    def test_inhiritence(self):
        # baselayer=mat.BaseLayer() 
        pass

        

if __name__ == '__main__':
    unittest.main()