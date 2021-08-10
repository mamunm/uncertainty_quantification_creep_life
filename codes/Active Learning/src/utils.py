import numpy as np 

def coverage(y, yL, yH):
    return (100 / y.shape[0] * ((y>yL)&(y<yH)).sum())