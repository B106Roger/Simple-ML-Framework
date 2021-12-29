import numpy as np
import _matrix as mat

myprint=print
def print(*args):
    myprint("PY:", *args)

in_feat=9
out_feat=3
batch=19
# input_tensor=np.random.uniform(low=-1, high=1, size=(batch, in_feat))
# input_tensor2=mat.Matrix(input_tensor)
input_tensor2=mat.Matrix(batch, in_feat)
layer=mat.Linear(in_feat, out_feat, False, True)
print(layer.get_weight())
layer_w1=layer.get_weight()[0]
layer_w1[0,0]=99
print(layer_w1.array)
layer_w1=layer_w1.T()
print(layer_w1.array)

print(input_tensor2.array.shape)
print(layer.m_weight.array.shape)
print(layer.m_bias.array.shape)

result=layer.forward(input_tensor2)
# process_input=input_tensor2.T()
# print('after construct process_input')
# result = mat.multiply_naive(layer.m_weight, process_input)
print(result.array.shape)
# print(result.array)
# print(result.array.shape)
# del layer
# del result
# del input_tensor2
sep='***********************'*5
print(f'{sep} python end here {sep}')

# del result
# print('deleted result')
# del input_tensor
# print('deleted input_tensor')
# del input_tensor2
# print('deleted input_tensor2')
# del layer