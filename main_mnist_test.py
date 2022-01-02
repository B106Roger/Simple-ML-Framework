import unittest
import numpy as np
import _matrix as mat
from tqdm import tqdm
from testcase.test_util import (
    print_matrix,
    init_matrix,
    print_ndarray,
    test_equal,
    AvgCounter
)


class TestStringMethods(unittest.TestCase):
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
        train_full=np.load('./testcase/train_data.npz')['train_data']
        train_label, train_data=train_full[:,:1], train_full[:,1:]
        test_full=np.load('./testcase/test_data.npz')['test_data']
        test_label, test_data=test_full[:,:1], test_full[:,1:]
        print(train_full.shape, test_full.shape)
        return

        for i, (in_feat, out_feat) in enumerate(layer_shape):
            # random_weight=np.random.randint(low=1, high=4, size=(in_feat, out_feat))
            random_weight=np.arange(1, 1+in_feat*out_feat).reshape(in_feat, out_feat)
            Mrandom_weight=mat.Matrix(random_weight)
            # random_bias=np.random.randint(low=1, high=4, size=(1, out_feat))
            random_bias=np.arange(1, 1+out_feat).reshape(1, out_feat)
            Mrandom_bias=mat.Matrix(random_bias)
            network.layers[i].set_weight((Mrandom_weight, Mrandom_bias))
            random_data=random_data@random_weight+random_bias
        
        batch_size=128
        for epoch in range(5):
            # for 
            Mresult=network(Mrandom_data)
            # print(f'Mrandom_data: {Mrandom_data.array}')
            print(f'Mresult: {Mresult.array}')
            loss=loss_fn(Mresult, Mgth)
            print(f'epoch {epoch:3d}: {loss.array.sum()}')
            print(f'loss gradient: {loss_fn.backward().array}')
            gradients=network.backward(loss_fn.backward())
            for i, (w_grad, b_grad) in enumerate(gradients):
                print(f'w_grad: {w_grad.array.flatten()}')
                print(f'b_grad: {b_grad.array.flatten()}')
                w_weight, b_weight=network.layers[i].get_weight()
                # print(f'w_weight: {w_weight.array.flatten()}')
                # print(f'b_weight: {b_weight.array.flatten()}')
                lr=0.00001
                w_weight = mat.Matrix(w_weight.array-w_grad.array*lr)
                b_weight = mat.Matrix(b_weight.array-b_grad.array*lr)
                # print(w_weight.array)
                # print(b_weight.array)
                network.layers[i].set_weight((w_weight, b_weight))
            print('-'*100)
        print('gradients', gradients)

    def test_activation_forward(self):
        relu=mat.ReLU()
        sigmoid=mat.Sigmoid()
        data=np.random.uniform(-1,1, size=(4,9))
        Mdata=mat.Matrix(data)

        res_relu=data.copy()
        res_relu[res_relu<0]=0
        test_res_relu=relu(Mdata)
        self.assertTrue(test_equal(test_res_relu, res_relu))

        res_sig=1.0/(1.0+np.exp(-data))
        test_res_sig=sigmoid(Mdata)
        self.assertTrue(test_equal(test_res_sig, res_sig))

    def test_mnist_mse(self):
        return
        a=[
            mat.Linear(784, 128, True, True),
            mat.Sigmoid(),
            mat.Linear(128, 10, True, True),
            mat.Sigmoid(),
        ]
        network=mat.Network(a)
        
        loss_fn=mat.MSE()

        # initialize dataset
        batch_size=300
        train_full=np.load('./testcase/train_data.npz')['train_data']
        train_label, train_data=train_full[:,0], train_full[:,1:]
        train_data = train_data / 255.0 - 0.5
        train_label = train_label.astype(np.int32)
        test_full=np.load('./testcase/test_data.npz')['test_data']
        test_label, test_data=test_full[:,0], test_full[:,1:]
        test_data = test_data / 255.0 - 0.5
        test_label = test_label.astype(np.int32)

        for layer in network.layers:
            config=layer.get_config()
            print(config)
            if 'in_dim' not in config.keys():
                continue
            in_feat = config['in_dim']
            out_feat = config['out_dim']
            random_weight=np.random.standard_normal((in_feat, out_feat))*0.5
            Mrandom_weight=mat.Matrix(random_weight)
            # random_bias=np.random.standard_normal((1, out_feat))*0.01 + 1 / out_feat
            random_bias=np.ones((1, out_feat)) + 1 / out_feat
            Mrandom_bias=mat.Matrix(random_bias)
            layer.set_weight((Mrandom_weight, Mrandom_bias))
        
        # epoch_counter=AvgCounter()
        opt=mat.SGD(1e-3, 0.0)
        for epoch in range(15):
            step_counter_loss=AvgCounter()
            step_counter_acc=AvgCounter()
            
            indices=np.arange(len(train_data))
            np.random.shuffle(indices)
            for st_idx in (range(0, len(train_full), batch_size)):
                train_indices = indices[st_idx: st_idx+batch_size]
                batch_train=train_data[train_indices]
                batch_label=train_label[train_indices]

                Mdata=mat.Matrix(batch_train)
                gth=np.zeros((batch_size, 10))
                gth[np.arange(batch_size),batch_label]=1

                Mgth =mat.Matrix(gth)

                Mresult=network(Mdata)
                loss=loss_fn(Mresult, Mgth)
                # loss=(batch, 10)
                step_counter_loss.update(loss.array.mean(axis=-1))
                step_counter_acc.update(Mresult.array.argmax(-1)==batch_label)
                print(f'\repoch {epoch:3d} step: {st_idx/batch_size+1} {step_counter_loss.total}/{len(train_full)}: Loss/Acc = {step_counter_loss.mean:6.4f}/{step_counter_acc.mean:6.4f}', end='')
                gradients=network.backward(loss_fn.backward())
                opt.apply_gradient(network, gradients)
            print()

    def test_mnist_crossentropy(self):
        a=[
            mat.Linear(784, 128, True, True),
            mat.Sigmoid(),
            mat.Linear(128, 10, True, True),

        ]
        network=mat.Network(a)
        
        loss_fn=mat.CategoricalCrossentropy()
        # loss_fn=mat.MSE()


        # initialize dataset
        batch_size=300
        train_full=np.load('./testcase/train_data.npz')['train_data']
        train_label, train_data=train_full[:,0], train_full[:,1:]
        train_data = train_data / 255.0 - 0.5
        train_label = train_label.astype(np.int32)
        test_full=np.load('./testcase/test_data.npz')['test_data']
        test_label, test_data=test_full[:,0], test_full[:,1:]
        test_data = test_data / 255.0 - 0.5
        test_label = test_label.astype(np.int32)

        for layer in network.layers:
            config=layer.get_config()
            print(config)
            if 'in_dim' not in config.keys():
                continue
            in_feat = config['in_dim']
            out_feat = config['out_dim']
            random_weight=np.random.standard_normal((in_feat, out_feat))*0.5
            Mrandom_weight=mat.Matrix(random_weight)
            # random_bias=np.random.standard_normal((1, out_feat))*0.01 + 1 / out_feat
            random_bias=np.ones((1, out_feat)) + 1 / out_feat
            Mrandom_bias=mat.Matrix(random_bias)
            layer.set_weight((Mrandom_weight, Mrandom_bias))
        
        # epoch_counter=AvgCounter()
        opt=mat.SGD(1e-3, 0.0)
        for epoch in range(15):
            step_counter_loss=AvgCounter()
            step_counter_acc=AvgCounter()
            
            indices=np.arange(len(train_data))
            np.random.shuffle(indices)
            for ii, st_idx in enumerate(range(0, len(train_full), batch_size)):
                train_indices = indices[st_idx: st_idx+batch_size]
                batch_train=train_data[train_indices]
                batch_label=train_label[train_indices]

                Mdata=mat.Matrix(batch_train)
                gth=np.zeros((batch_size, 10))
                gth[np.arange(batch_size),batch_label]=1

                Mgth =mat.Matrix(gth)

                Mresult=network(Mdata)
                loss=loss_fn(Mresult, Mgth)
                # loss=(batch, 10)
                step_counter_loss.update(loss.array.mean(axis=-1))
                step_counter_acc.update(Mresult.array.argmax(-1)==batch_label)
                print(f'\repoch {epoch:3d} step: {st_idx/batch_size+1} {step_counter_loss.total}/{len(train_full)}: Loss/Acc = {step_counter_loss.mean:6.4f}/{step_counter_acc.mean:6.4f}', end='')
                gradients=network.backward(loss_fn.backward())
                opt.apply_gradient(network, gradients)
            print()
if __name__ == '__main__':
    unittest.main()