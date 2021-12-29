import numpy as np
import _matrix as mat

a=np.arange(0,16).reshape(4,4)
b=mat.Matrix(a)
print(b.array)

c=b.T()
print(c.array)