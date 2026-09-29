# Code following the structure of ikdv and the note from Mats, Douglas, mm. 
# HSG 29.09

import numpy as np 
import numpy.fft as nf
import matplotlib.pyplot as plt 
from abc import ABC, abstractmethod

from linear.help import *
from nonlinear.solver import *
from nonlinear.spectralSolvers import * 

class WaveCollection(ABC):
    """Collection of different waves for LSM"""
    def __init__(self, P, N, dt, T, amp_list, mean_list):
        self.P  = P
        self.N  = N
        self.dx = P/N
        self.T  = T
        self.dt = dt

        self.k = 2*np.pi*nf.fftfreq(N, self.dx)

        self.amp_list = amp_list
        self.mean_list = mean_list

    @abstractmethod
    def wave_function(self, target_amp, mean):
        """Override with the travelling wave solver"""

    def make_travelling_waves(self):
        H0 = []

        for amp, mean in zip(self.amp_list, self.mean_list):
            eta = self.wave_function(target_amp=amp, mean=mean)
            eta.solve_w_continuation()
            H0.append(nf.irfft(eta.phi_hat, n=self.N))


class KdVLSM(WaveCollection):
    def wave_function(self, target_amp, mean):
        return KdVSolver(P=self.P, target_amp= target_amp, mean=mean, N = self.N)



if __name__ == '__main__':
    amp_l = [0.05,0.1]
    mean_l = [0.0,0.0]

    P = 2*np.pi
    N = 2**5
    dt = 5e-3
    T=2

    wpkdv = KdVLSM(P=P,N=N, dt=dt, T=T, amp_list=amp_l, mean_list=mean_l)
    wpkdv.make_travelling_waves()