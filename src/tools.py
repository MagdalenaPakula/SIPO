import numpy as np

def load(path):
    file = np.loadtxt(path, delimiter=",", quotechar='"', 
                      dtype=str, encoding='utf-8', skiprows=1)
    n = len(file[0])
    Nxn = [x if len(x) > 1 else x[0] for x in file[:, 0:n-1]]  
    N = file[:, n-1]

    return Nxn, N