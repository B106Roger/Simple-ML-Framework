from layers.dense import *
from layers.activation import *
from layers.loss import *
from layers.module import Module
from layers.optimizer import *
from helper import generate_linear, generate_XOR_easy, show_result, show_data
import h5py

# Prepare Model
with_bias=True
layers = [
    DenseLayer(2, 2, bias=with_bias),
    DenseLayer(2, 1, bias=with_bias),
]
mse = MSE()
model = Module(layers)
optimizer = SGD(lr=0.1, momentum=0.7)
#SGD(lr=0.8) # 1.0,400: 0.00506; 2.0, 400: 0.0009795
# optimizer = Adam(lr=0.1) # 1e-2,400: 0.02557; 1e-1, 400: 0.00069
model.set_weights([
    {'kernel':np.arange(1,5).reshape(2,2).astype(np.float32), 'bias':np.array([1,2]).astype(np.float32)},
    {'kernel':np.arange(1,3).reshape(2,1).astype(np.float32), 'bias':np.array([1]).astype(np.float32)}
])
# Prepare Data x=(batch, 2), y=(batch, 1)
x = np.arange(1,3).reshape(1,2)
y = np.zeros((1,1))
# x, y = generate_XOR_easy()
# x, y = generate_linear(n=100)


def train():
    losses = []
    precisions = []
    for epoch in range(4):
        # Gradient Descent
        result = model(x)
        print('result', result)
        # print(y.shape, result.shape)
        # exit()
        loss = mse(y, result)
        grads = model.backward(mse.backward())
        print(grads)
        optimizer.apply_gradients(model, grads)
        losses.append(loss.mean())

        # Compute Accuracy
        pred = model(x)
        final_pred = np.round(pred)
        total_count = len(x)
        correct_count = np.sum((y == final_pred).astype(np.float32))
        precision = correct_count/total_count
        precisions.append(precision)

        print(f'epoch: {epoch} loss: {loss.mean()} accuracy: {precision}')

    pred = model(x)
    final_pred = np.round(pred)
    print(final_pred[:10])
    show_result(x, y, final_pred, show_directly=False)
    show_data(losses, 'Training Loss', show_directly=False)
    show_data(precisions, 'Training Precision', show_directly=True)

train()
