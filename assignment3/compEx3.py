import numpy as np
from enforce_essential import enforce_essential

if __name__ == "__main__":
    E = np.load('variables/E.npy')
    enforce_essential(E)
    print(E.shape)