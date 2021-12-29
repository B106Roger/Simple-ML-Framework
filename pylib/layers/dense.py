from layers.base_layer import Layer
from layers.debug import *
import numpy as np
import h5py



class DenseLayer(Layer):
    def __init__(self, in_features, out_features, bias=False):
        super(DenseLayer, self).__init__(trainable=True, has_var=True, transpose_input=True)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # Variable Generation
        variance_var = 0.5 # np.sqrt(1.0/(in_features+out_features))
        self._trainable_vars_ = np.random.standard_normal((self.in_features, self.out_features)) * variance_var
        # self._trainable_vars_ = np.random.random((self.out_features, self.in_features))
        if self.bias:
            variance_bias = np.sqrt(1.0/out_features)
            self._trainable_bias_ = np.random.standard_normal((1, self.out_features)) * variance_bias

    def forward(self, input_tensor):
        d_print('---------------------------------- Forward')
        # d_print('self._trainable_vars_', self._trainable_vars_.shape)
        # d_print('input_tensor', input_tensor.shape)
        
        # input_tesnor=(batch, in_feat)
        # trainable_vars=(in_feat, out_feat)
        # result=(batch, out_feat)
        result = np.matmul(input_tensor, self._trainable_vars_)
        if self.bias:
            result = result + self._trainable_bias_

        d_print('variable', self._trainable_vars_)
        d_print('input', input_tensor)
        d_print('result',result.shape, result)
        return result
    
    def backward(self, input_grad=1.0):
        # input_grad=(batch, out_feature)
        # self._input_tensor_=(batch, in_feature)
        
        # trainable_grads=(in_feature, out_feature)
        trainable_grads = np.matmul(self._input_tensor_.T, input_grad)

        # self._trainable_vars_=(in_feature, out_feature)
        # input_grad =(batch, out_feature)
        # (in_feature, batch)
        dzda = np.matmul(input_grad, self._trainable_vars_.T)


        if self.bias:
            trainable_bias_grads = input_grad.sum(axis=0, keepdims=True)
            return [trainable_grads, trainable_bias_grads], dzda
        return [trainable_grads], dzda

    def apply_grad(self, grad):
        self._trainable_vars_ -= grad[0]
        if self.bias:
            self._trainable_bias_ -= grad[1]

    def set_weights(self, weights):
        for key in weights.keys():
            print(weights[key].shape)
        self._trainable_vars_ = weights['kernel']
        if self.bias:
            self._trainable_bias_ = np.expand_dims(weights['bias'], axis=0)

