import unittest
import numpy as np
from numpy import random
import _matrix as mat
from testcase.test_util import (
    print_matrix,
    init_matrix,
    print_ndarray,
    test_equal,
)


class TestStringMethods(unittest.TestCase):
    # def test_layer_construct(self):
    #     batch_size=1
    #     layer_shapes=[
    #         (16,32),
    #         (32,64),
    #         # (64,32),
    #         # (32,64),
    #         (64,128),
    #         (128,64),
    #     ]
    #     # Initialize dataset, weight
    #     random_data  =np.random.randint(low=-10, high=10, size=(batch_size, layer_shapes[0][0])).astype(np.float64)
    #     Mrandom_data  =mat.Matrix(random_data)
        
    #     for in_feat, out_feat in layer_shapes:
    #         random_weight=np.random.randint(low=-10, high=10, size=(in_feat, out_feat)).astype(np.float64)
    #         random_bias  =np.random.randint(low=-10, high=10, size=(1,       out_feat)).astype(np.float64)
    #         layer1=mat.Linear(in_feat, out_feat, True, True)
            
    #         # Transform to Matrix type
    #         Mrandom_weight=mat.Matrix(random_weight)
    #         Mrandom_bias  =mat.Matrix(random_bias)

    #         # set layer weight
    #         layer1.set_weight((Mrandom_weight, Mrandom_bias))

    #         # calculate result in C++ and numpy
    #         Mrandom_data=layer1(Mrandom_data)
    #         random_data=random_data@random_weight+random_bias
    #         print(Mrandom_data.array.shape, random_data.shape)

    #         # assert shape equal
    #         self.assertTrue(random_data.shape == Mrandom_data.array.shape)
    #         # assert content equal
    #         self.assertTrue(test_equal(Mrandom_data, random_data))

    #         # if test_equal(Mrandom_data, random_data):
    #         #     print(Mrandom_data.array)
    #         #     print('-'*50)
    #         #     print(random_data)

    # def test_network_construct(self):
    #     print('*'*70)
    #     batch=4
    #     layer_shape=[
    #         (16,32),
    #         (32,64),
    #         (64,128),
    #         (128,16)
    #     ]
    #     a=[
    #         mat.Linear(*layer_shape[0], True, True),
    #         mat.Linear(*layer_shape[1], True, True),
    #         mat.Linear(*layer_shape[2], True, True),
    #         mat.Linear(*layer_shape[3], True, True),
    #     ]
    #     network=mat.Network(a)
    #     # initialize network
    #     random_data=np.random.randint(low=-5,high=5,size=(batch, layer_shape[0][0]))
    #     Mrandom_data=mat.Matrix(random_data)
    #     result=network(Mrandom_data)
    #     self.assertTrue(result.array.shape==(batch, layer_shape[-1][-1]))

    #     for i, (in_feat, out_feat) in enumerate(layer_shape):
    #         random_weight=np.random.randint(low=-5, high=5, size=(in_feat, out_feat))
    #         Mrandom_weight=mat.Matrix(random_weight)
    #         random_bias=np.random.randint(low=-5, high=5, size=(1, out_feat))
    #         Mrandom_bias=mat.Matrix(random_bias)
    #         network.layers[i].set_weight((Mrandom_weight, Mrandom_bias))
    #         random_data=random_data@random_weight+random_bias
    #     Mrandom_data=network(Mrandom_data)

    #     self.assertTrue(test_equal(Mrandom_data, random_data))
    #     self.assertFalse(id(Mrandom_data)==id(random_data))
    #     print(hex(id(Mrandom_data)), hex(id(random_data)))
    #     print(Mrandom_data)

    # def test_loss_gradient_computing(self):
    #     # test loss computing
    #     random_data=np.random.randint(low=-5, high=5, size=(5,1))
    #     M_random_data=mat.Matrix(random_data)
    #     ground_truth=np.zeros(random_data.shape)
    #     M_ground_truth=mat.Matrix(ground_truth)

    #     mse=mat.MSE()
    #     test_loss=mse(M_random_data, M_ground_truth)
    #     test_gradient=mse.backward()

    #     result_loss=(random_data-ground_truth)**2
    #     result_grad=2*(random_data-ground_truth)
        
    #     self.assertTrue(test_equal(test_loss, result_loss))
    #     self.assertTrue(test_equal(test_gradient, result_grad))

    def test_model_backward(self):
        batch=1
        layer_shape=[
            (4,4),
            (4,1),
            # (16,32),
            # (32,64),
            # (64,4),
            # (64,128),
            # (128,4)
        ]
        a=[
            mat.Linear(*layer_shape[i], True, True)
            for i in range(len(layer_shape))
        ]
        network=mat.Network(a)
        loss_fn=mat.MSE()

        # initialize network
        # random_data=np.random.randint(low=-1,high=1,size=(batch, layer_shape[0][0]))
        random_data=np.arange(1, 1+layer_shape[0][0]).reshape(1, layer_shape[0][0])
        Mrandom_data=mat.Matrix(random_data)
        result=network(Mrandom_data)
        Mgth=mat.Matrix(np.zeros((batch, layer_shape[-1][-1]))*1.5)
        self.assertTrue(result.array.shape==(batch, layer_shape[-1][-1]))

        for i, (in_feat, out_feat) in enumerate(layer_shape):
            # random_weight=np.random.randint(low=1, high=4, size=(in_feat, out_feat))
            random_weight=np.arange(1, 1+in_feat*out_feat).reshape(in_feat, out_feat)
            Mrandom_weight=mat.Matrix(random_weight)
            # random_bias=np.random.randint(low=1, high=4, size=(1, out_feat))
            random_bias=np.arange(1, 1+out_feat).reshape(1, out_feat)
            Mrandom_bias=mat.Matrix(random_bias)
            network.layers[i].set_weight((Mrandom_weight, Mrandom_bias))
            random_data=random_data@random_weight+random_bias
        
        for epoch in range(5):
            Mresult=network(Mrandom_data)
            # print(f'Mrandom_data: {Mrandom_data.array}')
            print(f'Mresult: {Mresult.array}')
            loss=loss_fn(Mresult, Mgth)
            print(f'epoch {epoch:3d}: {loss.array.sum()}')
            print(f'loss gradient: {loss_fn.backward().array}')
            gradients=network.backward(loss_fn.backward())
            for i, (w_grad, b_grad) in enumerate(gradients):
                w_weight, b_weight=network.layers[i].get_weight()
                
                lr=0.00001
                w_weight = mat.Matrix(w_weight.array-w_grad.array*lr)
                b_weight = mat.Matrix(b_weight.array-b_grad.array*lr)
                
                network.layers[i].set_weight((w_weight, b_weight))
            print('-'*100)
        print('gradients', gradients)

    def test_optimizer(self):
        batch=1
        layer_shape=[
            (4,4),
            (4,1),
            # (16,32),
            # (32,64),
            # (64,4),
            # (64,128),
            # (128,4)
        ]
        a=[
            mat.Linear(*layer_shape[i], True, True)
            for i in range(len(layer_shape))
        ]
        network=mat.Network(a)
        opt=mat.SGD(0.00001, 0.0)
        loss_fn=mat.MSE()

        # initialize network
        # random_data=np.random.randint(low=-1,high=1,size=(batch, layer_shape[0][0]))
        random_data=np.arange(1, 1+layer_shape[0][0]).reshape(1, layer_shape[0][0])
        Mrandom_data=mat.Matrix(random_data)
        result=network(Mrandom_data)
        Mgth=mat.Matrix(np.zeros((batch, layer_shape[-1][-1]))*1.5)
        self.assertTrue(result.array.shape==(batch, layer_shape[-1][-1]))

        for i, (in_feat, out_feat) in enumerate(layer_shape):
            # random_weight=np.random.randint(low=1, high=4, size=(in_feat, out_feat))
            random_weight=np.arange(1, 1+in_feat*out_feat).reshape(in_feat, out_feat)
            Mrandom_weight=mat.Matrix(random_weight)
            # random_bias=np.random.randint(low=1, high=4, size=(1, out_feat))
            random_bias=np.arange(1, 1+out_feat).reshape(1, out_feat)
            Mrandom_bias=mat.Matrix(random_bias)
            network.layers[i].set_weight((Mrandom_weight, Mrandom_bias))
            random_data=random_data@random_weight+random_bias
        
        for epoch in range(5):
            Mresult=network(Mrandom_data)
            # print(f'Mrandom_data: {Mrandom_data.array}')
            print(f'Mresult: {Mresult.array}')
            loss=loss_fn(Mresult, Mgth)
            print(f'epoch {epoch:3d}: {loss.array.sum()}')
            print(f'loss gradient: {loss_fn.backward().array}')
            gradients=network.backward(loss_fn.backward())
            opt.apply_gradient(network, gradients)
            print('-'*100)
        print('gradients', gradients) 

if __name__ == '__main__':
    unittest.main()