# Solver for the general u_t + ikLu + 0.5ikN(u^2) formulation. 
# Solved with the StrangStep, and using same setup as in Spectral solver
# HSG 28.09

import numpy as np
import numpy.fft as nf
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from linear.help import *
from nonlinear.spectralSolvers import *

class Solver(ABC):
    def __init__(self, u0, P, dt, T, save_every:int = 1):
        self.u0 = u0
        self.u0_hat = nf.fft(u0)

        self.N  = len(u0)
        self.N2 = self.N//2
        self.Npad = 3*self.N//2
        self.P  = P
        self.dx = P/self.N

        self.k = nf.fftfreq(self.N, d=self.dx) * 2*np.pi
        self.mk = self.dispersion(self.k)

        self.dt = dt
        self.T = T
        self.save_every = save_every

        # Dealiasing by padding
        self.k_pad = 2*np.pi*nf.fftfreq(self.Npad, d = self.P/self.Npad)

    @abstractmethod
    def dispersion(self,k):
        """Return the Fourier symbol m(k) as an array"""

    def nonlinear_rhs(self, u_hat_in):
        u_hat_pad   = pad(u_hat_in, self.N)
        u_pad       = nf.ifft(u_hat_pad)*3/2
        nl_hat_pad  = 1j*self.k_pad*nf.fft(u_pad**2) * 2/3

        nl_hat = np.zeros(self.N, dtype=complex)
        nl_hat[:self.N2] = nl_hat_pad[:self.N2]
        nl_hat[-(self.N2-1):] = nl_hat_pad[-(self.N2-1):]
        rhs = -3*nl_hat
        return rhs

    def rk4_step(self, u_hat_in, h):
        """RK4"""
        k1 = self.nonlinear_rhs(u_hat_in)
        k2 = self.nonlinear_rhs(u_hat_in + 0.5*h*k1)
        k3 = self.nonlinear_rhs(u_hat_in + 0.5*h*k2) 
        k4 = self.nonlinear_rhs(u_hat_in + h*k3)
        
        return u_hat_in + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def splitStep(self):
        u_hat = self.u0_hat.copy()
        t_list = [0.0]
        u_list = [self.u0.copy()]

        t_current = 0.0
        step_count = 0

        E = np.exp(-1j*self.k*self.mk*self.dt / 2)
        # E2 = np.exp(-1j*self.k*self.mk*self.dt)
        
        while t_current < self.T - 1e-14:
            h = min(self.dt , self.T- t_current) #Truncating final step
            if h != self.dt:
                E = np.exp(-1j*self.k*self.mk*h/ 2)
                # E2 = np.exp(-1j*self.k*self.mk*h)

            u_hat = E*self.rk4_step(u_hat*E,h)

            t_current += h
            step_count += 1

            if step_count % self.save_every == 0 or t_current >= self.T - 1e-14:
                        t_list.append(t_current)
                        u_list.append(np.fft.ifft(u_hat).real)

        self.t, self.u = np.array(t_list), np.array(u_list)
        return np.array(t_list), np.array(u_list)

    def conservation(self):
            if not hasattr(self, "u"):
                print("Solving PDE for u")
                self.splitStep()
    
            u = self.u
            #Mass
            m = mass(u,self.dx)
            conservationInfo(m,"Mass")
            # L2
            l = l2(u,self.dx)
            conservationInfo(l, "L2")
            #Hamiltonian
            h = hamiltonian(u,self.mk,self.dx)
            conservationInfo(h, "Hamiltonian")

class KdV(Solver):
     def dispersion(self, k):
         return 1 - k**2

class Whitham(Solver):
     def dispersion(self, k):
        mk = np.ones_like(k)
        nz = k != 0
        mk[nz] = np.sqrt( np.tanh(k[nz]) / k[nz])
        return mk

class fKdV(Solver):
     def __init__(self, P, target_amp, mean, N, alpha):
             self.alpha = alpha
             super().__init__(P, target_amp, mean, N)
         
     def dispersion(self, k):        
        return 1 - np.abs(k)**self.alpha


if __name__ == '__main__':
     # Test of KdV solver
     N = 2**6
     P = 2*np.pi

     x = np.linspace(-P/2,P/2,N, endpoint=False)
     c = 4
     x0 = 0
     def soliton(x,c, x0):
             return (c/2) * 1/(np.cosh(np.sqrt(c)/2 *(x - x0))**2)
     
     u0 = soliton(x,c,x0)

     kdv = KdV(u0, P,dt= 5e-4, T=1)
     t, u = kdv.splitStep()

     animate_solution(x,u)