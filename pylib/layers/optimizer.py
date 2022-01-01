import numpy as np


class SGD:
    def __init__(self, lr=1e-2, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

        self._previous_grads_ = {}
    
    def process_gradient(self, grads):
        new_grads = []
        for layer_idx in range(len(grads)):
            layer_grads = []
            for var_idx in range(len(grads[layer_idx])):
                current_grad = grads[layer_idx][var_idx]
                unique_key = f'{layer_idx}_{var_idx}'
                if self.momentum != 0.0:
                    previous_grad = self._previous_grads_.get(unique_key, 0.0)
                    new_var_grad = current_grad * self.lr + previous_grad * self.momentum
                else:
                    new_var_grad = current_grad * self.lr
                layer_grads.append(new_var_grad)
                self._previous_grads_[unique_key] = new_var_grad
            new_grads.append(layer_grads)
        return new_grads

    def apply_gradients(self, model, grads):
        new_grads = self.process_gradient(grads)
        model.apply_grads(new_grads)
        return

class SGD_back:
    def __init__(self, lr=1e-2, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

        self._previous_grads_ = None
    
    def process_gradient(self, grads):
        new_grads = []
        for layer_idx in range(len(grads)):
            layer_grads = []
            for var_idx in range(len(grads[layer_idx])):
                current_grad = grads[layer_idx][var_idx]
                if self._previous_grads_ is not None and self.momentum != 0.0:
                    previous_grad = self._previous_grads_[layer_idx][var_idx]
                    layer_grads.append(current_grad * self.lr + previous_grad * self.momentum)
                else:
                    layer_grads.append(current_grad * self.lr)
            new_grads.append(layer_grads)
        return new_grads

    def apply_gradients(self, model, grads):
        new_grads = self.process_gradient(grads)
        model.apply_grads(new_grads)
        self._previous_grads_=new_grads
        return

class AdaGrad:
    def __init__(self, lr=1e-2, initial_accumulator_value=0.1, momentum=0.0, epsilone=1e-7):
        self.lr = lr
        self.momentum = momentum

        self.epsilone = epsilone
        self.initial_accumulator_value = initial_accumulator_value
        self._previous_grads_ = {}
        self._accumulator_ = {}
    
    def process_gradient(self, grads):
        new_grads = []
        for layer_idx in range(len(grads)):
            layer_grads = []
            for var_idx in range(len(grads[layer_idx])):
                current_grad = grads[layer_idx][var_idx]
                unique_key = f'{layer_idx}_{var_idx}'
                reg = self._accumulator_.get(unique_key, self.initial_accumulator_value) + np.power(current_grad, 2.0)
                self._accumulator_[unique_key] = reg
                if self.momentum != 0.0:
                    previous_grad = self._previous_grads_.get(unique_key, 0.0)
                    new_var_grad = current_grad * self.lr / np.power(reg + self.epsilone, 0.5) + previous_grad * self.momentum
                else:
                    new_var_grad = current_grad * self.lr / np.power(reg + self.epsilone, 0.5)

                layer_grads.append(new_var_grad)
                self._previous_grads_[unique_key] = new_var_grad
            new_grads.append(layer_grads)
        return new_grads

    def apply_gradients(self, model, grads):
        new_grads = self.process_gradient(grads)
        model.apply_grads(new_grads)
        return

class AdaDelta:
    """
    Reference:
    https://ckmarkoh.github.io/blog/2016/02/08/optimization-method-adadelta/
    https://stackoverflow.com/questions/56730888/what-is-the-learning-rate-parameter-in-adadelta-optimiser-for-in-keras
    https://github.com/keras-team/keras/blob/1.2.2/keras/optimizers.py
    """
    def __init__(self, lr=1e-2, rho=0.95, epsilon=1e-7):
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon
        self._previous_grads_ = {}
        self._accumulated_grads_ = {}
        self._delta_accumulated_grads_ = {}
    
    def process_gradient(self, grads):
        new_grads = []
        for layer_idx in range(len(grads)):
            layer_grads = []
            for var_idx in range(len(grads[layer_idx])):
                current_grad = grads[layer_idx][var_idx]
                unique_key = f'{layer_idx}_{var_idx}'
                # Update Regularizer
                accumulated = self._accumulated_grads_.get(unique_key, 0.0) * self.rho + \
                    np.power(current_grad, 2.0) * (1.0 - self.rho)
                self._accumulated_grads_[unique_key] = accumulated

                # Update Gradient and momentum
                new_var_grad = np.power(
                    (self._delta_accumulated_grads_.get(unique_key, 0.0) + self.epsilon) / (self._accumulated_grads_[unique_key] + self.epsilon), 
                    0.5) * current_grad
                    
                # Update Unit Correct Term
                self._delta_accumulated_grads_[unique_key] = self.rho * self._delta_accumulated_grads_.get(unique_key, 0.0) + (1.0 - self.rho) * np.power(new_var_grad, 2.0)

                layer_grads.append(new_var_grad * self.lr)
                self._previous_grads_[unique_key] = new_var_grad
            new_grads.append(layer_grads)
        return new_grads

    def apply_gradients(self, model, grads):
        new_grads = self.process_gradient(grads)
        model.apply_grads(new_grads)
        return

class RMSprop:
    # Reference:
    # tensorflow 2.2.2 doc: tensorflow/python/keras/optimizer_v1/rmsprop.py
    def __init__(self, lr=1e-3, rho=0.9, momentum=0.0, epsilon=1e-7):
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon
        self.momentum = momentum
        self._previous_grads_ = {}
        self._accumulated_grads_ = {}
    
    def process_gradient(self, grads):
        new_grads = []
        for layer_idx in range(len(grads)):
            layer_grads = []
            for var_idx in range(len(grads[layer_idx])):
                current_grad = grads[layer_idx][var_idx]
                unique_key = f'{layer_idx}_{var_idx}'
                # Update Regularizer
                accumulated = self._accumulated_grads_.get(unique_key, 0.0) * self.rho + \
                    np.power(current_grad, 2.0) * (1.0 - self.rho)
                self._accumulated_grads_[unique_key] = accumulated

                # Update Gradient and momentum
                new_var_grad = current_grad / (np.power(self._accumulated_grads_[unique_key] + self.epsilon, 0.5))
                if self.momentum != 0.0:
                    new_var_grad += self.momentum * self._previous_grads_.get(unique_key, 0.0) 

                layer_grads.append(new_var_grad* self.lr)
                self._previous_grads_[unique_key] = new_var_grad
            new_grads.append(layer_grads)
        return new_grads

    def apply_gradients(self, model, grads):
        new_grads = self.process_gradient(grads)
        model.apply_grads(new_grads)
        return

class Adam:
    def __init__(self, lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self._previous_grads_ = {}
        self._accumulated_grads_1_ = {}
        self._accumulated_grads_2_ = {}
    
    def process_gradient(self, grads):
        new_grads = []
        for layer_idx in range(len(grads)):
            layer_grads = []
            for var_idx in range(len(grads[layer_idx])):
                current_grad = grads[layer_idx][var_idx]
                current_grad2 = np.power(current_grad, 2.0)
                unique_key = f'{layer_idx}_{var_idx}'
                # Update Regularizer

                self._accumulated_grads_1_[unique_key] = self._accumulated_grads_1_.get(unique_key, current_grad) * self.beta_1 + \
                    current_grad * (1.0 - self.beta_1)
                self._accumulated_grads_2_[unique_key] = self._accumulated_grads_2_.get(unique_key, current_grad2) * self.beta_2 + \
                    current_grad2 * (1.0 - self.beta_2)

                # Update Gradient and momentum
                new_var_grad = self.lr * self._accumulated_grads_1_[unique_key] / (np.power(self._accumulated_grads_2_[unique_key], 0.5) + self.epsilon)

                layer_grads.append(new_var_grad)
                self._previous_grads_[unique_key] = new_var_grad
            new_grads.append(layer_grads)
        return new_grads

    def apply_gradients(self, model, grads):
        new_grads = self.process_gradient(grads)
        model.apply_grads(new_grads)
        return