# Creating a class for making solitons travelling or time dependent
# HSG
# 24.09

import numpy as np 
import numpy.fft as nf
import matplotlib.pyplot as plt 

class Soliton():
    def __init__(self, c, x0=0.0):
        self.c  = c
        self.x0 = x0
        
    def wave(self, t,x):
        t = np.atleast_1d(t)[:,None]
        x = np.atleast_1d(x)[None,:]
        self.eta = (self.c/2) * 1/(np.cosh(np.sqrt(self.c)/2 *(x- self.c*t - self.x0))**2)
    
    
# ---------------------
if __name__ == '__main__':
    # c_list = np.array([0.5,1])
    # eta_list = []
    # for c in c_list:
    #     eta_list.append(Soliton(c))
    
    c = 1.0
    N = 2**4
    L = 2*np.pi
    T = 1
    x_array = np.linspace(-L/2, L/2 , N, endpoint=False)
    t_array = np.linspace(0,T,4)
    
    wave = Soliton(c)
    wave.wave(t_array,x_array)
    
    print(wave.eta.shape)
    
    eta_hatx = np.fft.fft(wave.eta, axis=1)
    k = 2*np.pi*np.fft.fftfreq(N, d=(x_array[1]-x_array[0]))
    w = 2*np.pi*np.fft.fftfreq(4, d=(t_array[1]-t_array[0]))

    
    deta_dt_hat = nf.ifft(1j*w[:,None] * nf.fft(eta_hatx , axis = 0), axis = 0)
    print(deta_dt_hat.shape)
    # order = np.argsort(k)
    # plt.plot(k[order],eta_hatx[1,order])
    # plt.show()
    # for eta in eta_list:
    #     eta.wave(t,x_array)
        
    # for eta in eta_list:
    #     plt.plot(x_array,eta.eta[0,:] , label = f"c:{eta.c}")
    # plt.legend()
    # plt.show()