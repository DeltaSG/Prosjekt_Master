# implementing the cnoidal wave for comparison 

import numpy as np 
import matplotlib.pyplot as plt
from scipy.special import ellipj

def cnoidal(x,t,c,H,L,mean,m):
    """The cnoidal wave
    Inputs:
    x: array - the spacial domain
    t: array - the temporal domain
    c: float - the wave speed
    H: float - the wave amplitude
    L: float - the horizontal scale of the wave
    mean: float - the background level
    m: float - elliptic integral prower
    
    Output:
    eta(t,x): array - the cnoidal wave defined by the parameters"""
    X,T = np.meshgrid(x,t)
    u = (X-c*T)/L
    sn,cn,dn,ph = ellipj(u,m)
    eta = H*cn**2 + mean
    return eta

if __name__ == '__main__':
    L = 2
    mean = 0
    c = 1
    m = 0.9
    H=0.05

    b = 10
    x = np.linspace(-b/2, b/2 , 100)
    t = 0

    eta = cnoidal(x,t,c,H,L,mean,m)

    plt.plot(x,eta[0,:])
    plt.show()