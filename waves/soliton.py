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

import numpy as np
from scipy.special import ellipj, ellipk, ellipe
from scipy.optimize import brentq


def kdv_cnoidal(P, crest, mean,x,t):
    """
    Cnoidal solution for the KdV traveling-wave equation

        (1/6) phi_xx + (1-c) phi + phi**2/2 = B

    Assumes phi(0) = crest and mean(phi) = mean.
    """

    def parameters(m):
        K = ellipk(m)
        E = ellipe(m)

        kappa = 2 * K / P
        b = 2 * m * kappa**2

        # Mean of cn(u|m)**2 over one period
        q = (E - (1 - m) * K) / (m * K)

        # phi = a + b*cn**2
        a = mean - b * q

        return kappa, a, b, q

    def amplitude_residual(m):
        kappa, a, b, q = parameters(m)
        return a + b - crest

    # Find m from period, mean and crest value
    m = brentq(
        amplitude_residual,
        1e-7,
        1 - 1e-7
    )
    

    kappa, a, b, q = parameters(m)
    c = 1 + a + (2 / 3) * kappa**2 * (2 * m - 1)
    X,T = np.meshgrid(x,t)
    _, cn, _, _ = ellipj(kappa*(X - c*T), m)

    phi = a + b * cn**2

    return phi, c, m

def soliton(x,A, x0):
     kappa = np.sqrt(A/2)
     return A * 1/(np.cosh(kappa*(x - x0))**2)

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