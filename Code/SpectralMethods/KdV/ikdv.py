# full inverse KdV time ind/dep
# HSG
# 24.09

import numpy as np 
import numpy.fft as nf
import matplotlib.pyplot as plt 
from KdV.soliton import Soliton
from linear.help import make_x_array

class Packet():
    """Collection of waves for LSM"""
    def __init__(self,L,N,T,M, c_list, x0_list):
        self.c_list  = c_list
        self.x0_list = x0_list
        
        self.x       = make_x_array(N, L, symmetric=True)
        self.t       = make_x_array(M,T, symmetric=False)
        
        self.k = 2*np.pi*np.fft.fftfreq(N, d=(self.x[1]-self.x[0]))
        self.w = 2*np.pi*np.fft.fftfreq(M, d=(self.t[1]-self.t[0]))
        
        return

        
    def make_solitons(self):
        phi_hat_list    = []
        nl_hat_list     = []
        dHdt_hat_list   = []
        
        for c,x0  in zip(self.c_list, self.x0_list):
            eta = Soliton(c,x0).wave(self.t, self.x).eta
            eta_hat  = nf.fft(eta,      axis=1)
            eta2_hat = nf.fft(eta**2,   axis=1)
            detadt_hat = nf.ifft(
                1j*self.w[:,None] * nf.fft(
                    eta_hat , axis = 0),
                axis = 0)
            phi_hat_list.append(eta_hat)
            nl_hat_list.append(eta2_hat)
            dHdt_hat_list.append(detadt_hat)
            
        self.H      = np.stack(phi_hat_list,    axis = 0)
        self.H2     = np.stack(nl_hat_list,     axis = 0)
        self.dHdt   = np.stack(dHdt_hat_list,   axis = 0)
        
        return 

    def LSM(self):
        
        def to_freq_first(arr):
            """Move frequency to axis 0,
            and flatten (soliton, time) into one sample axis"""
            arr = np.moveaxis(arr, -1,0)
            return arr.reshape(arr.shape[0], -1) 
        
        Hr = to_freq_first(self.H)
        H2r = to_freq_first(self.H2)
        dHdtr = to_freq_first(self.dHdt)
        
        A = np.stack(
            [Hr, H2r],
            axis =1
        )
        A_pinv = np.linalg.pinv(A)
        X = np.einsum('fs,fsc->fc', dHdtr, A_pinv)
        
        self.ell = X[:,0]
        self.n   = X[:,1]
        
        return X
        