# Script for creating a set ow waves using the Newton sovler, and then the pde solver
# returning the format required for the lsm.

# building taller waves, one can use the previous solution (as in continuation) but across waves too

import numpy as np
import numpy.fft as nf
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from camera.camera_file import Camera
from camera.spectralCamera import *
from camera.pdeCamera import *
from camera.lsmCamera import LSM

class WavePacket(ABC):
    def __init__(self, P_list, amp_list, mean_list, camera:Camera, T, save_every:int = 1):
        self.Ps = P_list
        self.amps = amp_list
        self.means = mean_list
        self.camera = camera
        self.T = T
        self.save_every = save_every
        
    @abstractmethod
    def waveProfile(self,P, amp, mean):
        """Override with the Newton solver of the wave"""
        
    @abstractmethod
    def waveSolver(self, u0):
        """Override with the pde solver of the wave"""
        
    def mainloop(self):
        amp = None
        
        H = []
        
        for i in range(len(self.amps)):
            wavep = self.waveProfile(P = self.Ps[i], amp=self.amps[i] , mean=self.means[i])
            wavep.solve_w_continuation()
            phi0 = wavep.get_physical(wavep.phi_hat)
            twave = self.waveSolver(u0= phi0)
            # _, phi = twave.splitStep()
            _, phi = twave.ifrk4()
            
            H.append(phi)
        
        self.H = H
        return H
    
    
class KdVPacket(WavePacket):
    def waveProfile(self,P, amp, mean):
        return KdVNewton(P=P, target_amp=amp, mean = mean,camera=self.camera)
    
    def waveSolver(self, u0):
        return KdV(camera=self.camera, T = self.T, u0=u0,save_every=self.save_every)

class fKdVPacket(WavePacket):
    def __init__(self, P, N, amp_list, mean_list, dt, T, alpha):
        self.alpha = alpha
        super().__init__(P, N, amp_list, mean_list, dt, T)

    def waveProfile(self, P,amp, mean):
        return fKdVNewton(P=P, target_amp=amp, mean = mean,camera=self.camera, alpha=self.alpha)

    def waveSolver(self, u0):
        return fKdV( camera=self.camera, T = self.T, u0=u0, alpha=self.alpha, save_every=self.save_every)

class WhithamPacket(WavePacket):
    def waveProfile(self,P, amp, mean):
        return WhithamNewton(P=P, target_amp=amp, mean = mean, camera=self.camera)
    
    def waveSolver(self, u0):
        return Whitham(camera=self.camera, T = self.T, u0=u0, save_every=self.save_every)