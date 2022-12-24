import numpy as np
from layers.base_layer import Layer


def relu_forward(x):
    return np.maximum(x, 0.0)


def relu_backward(input_tensor, input_grads):
    grads_mask = (input_tensor > 0.0).astype(np.float32)
    input_grads = input_grads * grads_mask
    return input_grads


def sigmiod_forward(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmiod_backward(input_tensor, input_grads):
    sig = sigmiod_forward(input_tensor)
    return sig*sig*np.exp(-input_tensor)*input_grads

class ReLU(Layer):
    def __init__(self):
        super(ReLU, self).__init__(trainable=False, has_var=False, transpose_input=False)
        self.has_var = False
        self.trainable = False

    def forward(self, x):
        # return np.maximum(x, 0.0)
        return relu_forward(x)

    def backward(self, input_grads):
        trainable_grads = np.array([])

        # grads_mask = (self._input_tensor_ > 0.0).astype(np.float32)
        # input_grads = input_grads * grads_mask

        input_grads = relu_backward(self._input_tensor_, input_grads)
        return trainable_grads, input_grads
    
    def apply_grad(self, grad):
        raise RuntimeError('The ReLU function has no trainable var, therefore no need for apply gradient.')
        return
    def set_weights(self, weights):
        raise RuntimeError('The ReLU function has no trainable var, therefore no need for set weights.')
        return


def sigmiod_forward(x):
    return 1.0 / (1.0 + np.exp(-x))
def sigmiod_backward(input_tensor, input_grads):
    sig = sigmiod_forward(input_tensor)
    return sig*sig*np.exp(-input_tensor)*input_grads
class Sigmoid(Layer):
    def __init__(self):
        super(Sigmoid, self).__init__(trainable=False, has_var=False, transpose_input=False)
        self.has_var = False
        self.trainable = False

    def forward(self, x):
        return sigmiod_forward(x)

    def backward(self, input_grads):
        trainable_grads = np.array([])
        input_grads = sigmiod_backward(self._input_tensor_, input_grads)
        return trainable_grads, input_grads
    
    def apply_grad(self, grad):
        raise RuntimeError('The ReLU function has no trainable var, therefore no need for apply gradient.')
        return
    def set_weights(self, weights):
        raise RuntimeError('The ReLU function has no trainable var, therefore no need for set weights.')
        return

