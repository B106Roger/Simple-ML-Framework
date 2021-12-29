import _matrix as mat

a=mat.Matrix(4,4)
for i in range(4):
    a[i,i]=i
b=mat.Matrix(1,4)
for i in range(4):
    b[0,i]=i

print('test row broadcast')
print('a:\n', a.array)
print('b:\n', b.array)
c=a+b
print('c:\n',c.array)


a=mat.Matrix(4,4)
for i in range(4):
    a[i,i]=i
b=mat.Matrix(4,1)
for i in range(4):
    b[i,0]=i

print('test col broadcast')
print('a:\n', a.array)
print('b:\n', b.array)
c=a+b
print('c:\n',c.array)