import numpy as np
from layers.debug import *

class Loss:
    def __init__(self):
        pass
    def __call__(self, ground_truth, input_tensor):
        # ground_truth=(batch, dim)
        # input_tensor=(batch, dim)
        self.record_grads(ground_truth, input_tensor)
        x = self.forward(ground_truth, input_tensor)
        return x    
    def record_grads(self, ground_truth, input_tensor):
        self._input_tensor_ = input_tensor
        self._ground_truth_ = ground_truth

class MSE(Loss):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, ground_truth, input_tensor):
        result = np.power(ground_truth-input_tensor, 2.0)
        return result

    def backward(self):
        batch_grads = 2 * (self._input_tensor_ - self._ground_truth_)
        # batch_grads=(batch, dim)
        return batch_grads
