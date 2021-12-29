
def print_matrix(mat):
    r=mat.nrow
    c=mat.ncol
    for i in range(r):
        for j in range(c):
            print(f'{mat[i,j]:10.4f}', end=' ')
        print()

def init_matrix(mat):
    r=mat.nrow
    c=mat.ncol
    for i in range(r):
        for j in range(c):
            mat[i,j]=i*mat.ncol+j

def print_ndarray(ndarray):
    r=ndarray.shape[0]
    c=ndarray.shape[1]
    for i in range(r):
        for j in range(c):
            print(f'{ndarray[i,j]:10.4f}', end=' ')
        print()

def test_equal(mat1, ndarray):
    col=mat1.ncol
    row=mat1.nrow
    for i in range(row):
        for j in range(col):
            # print(f'mat1[{i},{j}]={mat1[i,j]} ndarray[{i},{j}]={ndarray[i,j]}')
            if mat1[i,j] != ndarray[i,j]: 
                print(f'wrong index: i: {i} j: {j}')
                return False
    return True
