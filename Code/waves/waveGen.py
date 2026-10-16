# Script for creating a set ow waves using the Newton sovler, and then the pde solver
# returning the format required for the lsm.

# building taller waves, one can use the previous solution (as in continuation) but across waves too

import numpy as np
import numpy.fft as nf
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from solvers.spectral import *
from solvers.pde import *
from solvers.help import *

class WavePacket(ABC):
    def __init__(self, P, N, amp_list, mean_list, dt, T):
        self.P = P
        self.N = N
        self.amps = amp_list
        self.means = mean_list
        
        self.dt = dt
        self.T = T
        
    @abstractmethod
    def waveProfile(self, amp, mean):
        """Override with the Newton solver of the wave"""
        
    @abstractmethod
    def waveSolver(self, u0):
        """Override with the pde solver of the wave"""
        
    def mainloop(self):
        u = None
        amp = None
        
        H = []
        
        for i in range(len(self.amps)):
            wavep = self.waveProfile(amp=self.amps[i] , mean=self.means[i])
            wavep.solve_w_continuation(u0= u, amp0 = amp)
            u = wavep.u
            amp = self.amps[i]
            phi0 = nf.irfft(wavep.phi_hat)
            twave = self.waveSolver(u0= phi0)
            _, phi = twave.splitStep()
            
            H.append(phi)
        
        self.H = H
        return H
    
    
class KdVPacket(WavePacket):
    def waveProfile(self, amp, mean):
        return KdVNewton(P=self.P, target_amp=amp, mean = mean, N=self.N)
    
    def waveSolver(self, u0):
        return KdV(P=self.P, dt=self.dt, T = self.T, u0=u0)

class fKdVPacket(WavePacket):
    def __init__(self, P, N, amp_list, mean_list, dt, T, alpha):
        self.alpha = alpha
        super().__init__(P, N, amp_list, mean_list, dt, T)

    def waveProfile(self, amp, mean):
        return fKdVNewton(P=self.P, target_amp=amp, mean = mean, N=self.N, alpha=self.alpha)

    def waveSolver(self, u0):
        return fKdV(P=self.P, dt=self.dt, T = self.T, u0=u0, alpha=self.alpha)

class WhithamPacket(WavePacket):
    def waveProfile(self, amp, mean):
        return WhithamNewton(P=self.P, target_amp=amp, mean = mean, N=self.N)
    
    def waveSolver(self, u0):
        return Whitham(P=self.P, dt=self.dt, T = self.T, u0=u0)