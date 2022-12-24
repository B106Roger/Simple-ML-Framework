import numpy as np


def generate_linear(n=100):
    import numpy as np
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    import numpy as np
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, pred_y, show_directly=True):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    if show_directly:
        plt.show()

def show_data(data, title='', show_directly=True):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(data)
    plt.title(title, fontsize=18)
    if show_directly:
        plt.show()

################################################
def generate_XOR_very_easy():
    x = np.array([
        [0.01, 0.01],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0]
    ])
    y = np.array([[1.0], [1.0], [0.0], [0.0]])
    return x,y