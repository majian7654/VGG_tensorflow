import numpy as np

def loadnpz(filename):
    arr = np.load(filename)
    return arr

if __name__=='__main__':
    weight = loadnpz('./vgg16_weights.npz')
    print(weight[0])
