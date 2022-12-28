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
# mat.set_matrix_mode(1)

# initialize dataset
EPOCH=15
batch_size=300
train_full=np.load('./testcase/train_data.npz')['train_data']
train_label, train_data=train_full[:,0], train_full[:,1:]
train_data = train_data / 255.0 - 0.5
train_label = train_label.astype(np.int32)
test_full=np.load('./testcase/test_data.npz')['test_data']
test_label, test_data=test_full[:,0], test_full[:,1:]
test_data = test_data / 255.0 - 0.5
test_label = test_label.astype(np.int32)

def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10
    return 1

def test_mnist_mse():
    ################################
    # Create Network, Loss Function
    ################################
    a=[
        mat.Linear(784, 128, True, True),
        mat.Sigmoid(),
        mat.Linear(128, 10, True, True),
        mat.Sigmoid(),
    ]
    network=mat.Network(a)
    loss_fn=mat.MSE()
    
    ################################
    # Init Network Parameter
    ################################
    for layer in network.layers:
        config=layer.get_config()
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
        
    ##################################
    # Determine Optmization Algorithm
    ##################################
    opt=mat.SGD(1e-3, 0.0)
    acc_list=[]
    for epoch in range(EPOCH):
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

            ##################################
            # Update Model Parameter
            ##################################
            Mresult=network(Mdata)
            loss=loss_fn(Mresult, Mgth)
            gradients=network.backward(loss_fn.backward())
            opt.apply_gradient(network, gradients)
            # loss=(batch, 10)
            step_counter_loss.update(loss.array.mean(axis=-1))
            step_counter_acc.update(Mresult.array.argmax(-1)==batch_label)
            print(f'\repoch {epoch:3d} {step_counter_loss.total}/{len(train_full)}: Loss/Acc = {step_counter_loss.mean:6.4f}/{step_counter_acc.mean:6.4f}', end='')
        acc_list.append(step_counter_acc.mean)
        print()
    return acc_list

def test_mnist_crossentropy():
    a=[
        mat.Linear(784, 128, True, True),
        mat.Sigmoid(),
        mat.Linear(128, 10, True, True),
    ]
    network=mat.Network(a)
    loss_fn=mat.CategoricalCrossentropy()

    for layer in network.layers:
        config=layer.get_config()
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
    acc_list=[]
    for epoch in range(EPOCH):
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
            gradients=network.backward(loss_fn.backward())
            opt.apply_gradient(network, gradients)
            # loss=(batch, 10)
            step_counter_loss.update(loss.array.mean(axis=-1))
            step_counter_acc.update(Mresult.array.argmax(-1)==batch_label)
            print(f'\repoch {epoch:3d} {step_counter_loss.total}/{len(train_full)}: Loss/Acc = {step_counter_loss.mean:6.4f}/{step_counter_acc.mean:6.4f}', end='')
        acc_list.append(step_counter_acc.mean)
        print()
        for i in range(128):
            batch_data=train_data[i:i+1]
            Mdata=mat.Matrix(batch_data)
            Mresult=network(Mdata)
            result=Mresult.array.argmax(-1)

            image=((batch_data.reshape(28,28,1)+0.5)*255.0).astype(np.uint8).repeat(3, axis=-1)
            cv2.rectangle(image, (0,0), (8,8), color=(255,255,255), thickness=-1)
            mytext=f'{result[0]}'
            cv2.putText(image, mytext, (2,8), cv2.FONT_HERSHEY_SIMPLEX, fontScale=get_optimal_font_scale(mytext, 8), color=(0,0,255))
            cv2.imwrite(f"visualize/{i}.png", image[...,::-1])
    return acc_list

if __name__ == '__main__':
    a=test_mnist_crossentropy()
    b=test_mnist_mse()
    plt.plot(a,label='CrossEntropy')
    plt.plot(b,label='MSE')
    plt.xlim(0, EPOCH-1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc="upper left")
    plt.savefig('result.png')