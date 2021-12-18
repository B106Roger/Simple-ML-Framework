from layers.dense import *
from layers.activation import *
from layers.loss import *
from layers.module import Module
from layers.optimizer import *
from helper import generate_linear, generate_XOR_easy, show_result, show_data
import h5py

# Prepare Model
with_bias=False
layers = [
    DenseLayer(2, 6, bias=with_bias),
    Sigmoid(),
    DenseLayer(6, 6, bias=with_bias),
    Sigmoid(),
    DenseLayer(6, 1, bias=with_bias),
    Sigmoid(),

]
mse = MSE()
model = Module(layers)
optimizer = SGD(lr=0.1, momentum=0.7)
#SGD(lr=0.8) # 1.0,400: 0.00506; 2.0, 400: 0.0009795
# optimizer = Adam(lr=0.1) # 1e-2,400: 0.02557; 1e-1, 400: 0.00069

# Prepare Data x=(batch, 2), y=(batch, 1)
x, y = generate_XOR_easy()
# x, y = generate_linear(n=100)


def train():
    losses = []
    precisions = []
    for epoch in range(1000):
        # Gradient Descent
        result = model(x)
        loss = mse(y, result)
        grads = model.backward(mse.backward())
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

# train()
