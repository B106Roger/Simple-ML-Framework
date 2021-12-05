import h5py
import numpy as np

class Module:
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def backward(self, grad):
        grads = []
        accumulate_grads = grad

        for layer in reversed(self.layers):
            tranable_grads, accumulate_grads = layer.backward(accumulate_grads)
            # grads.append(tranable_grads)
            if layer.has_var and layer.trainable:
                grads.append(tranable_grads)
        return [grad for grad in reversed(grads)]

    def apply_grads(self, grads):
        grad_layers = list(filter(lambda layer: layer.has_var and layer.trainable, self.layers))
        for layer, grad in zip(grad_layers, grads):
            lr_grad = [g for g in grad]
            layer.apply_grad(lr_grad)

    def set_weights(self, weights):
        var_layers = list(filter(lambda layer: layer.has_var, self.layers))
        for layer, weights in zip(var_layers, weights):
            layer.set_weights(weights)


    def load_weights(self, filename):
        hf = h5py.File(filename, 'r')
        grads = []
        for layer_key in hf.keys():
            if (layer_key.find('input') != -1):
                continue
            layer_hf = hf.get(layer_key)
            layer_hf = layer_hf.get(list(layer_hf.keys())[0])
            layer_grad = {}
            for var_key in layer_hf.keys():
                var_grad = np.array(layer_hf.get(var_key)).T
                if 'bias' in var_key:
                    layer_grad['bias'] = var_grad
                elif 'kernel' in var_key:
                    layer_grad['kernel'] = var_grad
            print(len(layer_grad))
            grads.append(layer_grad)


        self.set_weights(grads)
