# This file shall contain the numerical pde solvers of the waves

# Solver for the general u_t + ikLu + 0.5ikN(u^2) formulation. 
# Solved with the StrangStep, and using same setup as in Spectral solver
# HSG 28.09

import numpy as np
import numpy.fft as nf
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from solvers.spectral import *
from solvers.help import *

class Solver(ABC):
    def __init__(self, P, dt, T,u0 = None, u0_hat = None, save_every:int = 1):
        if u0 is not None:
            self.u0 = u0
            self.u0_hat = nf.fft(u0)
        elif u0_hat is not None:
            self.u0_hat = u0_hat
            self.u0 = nf.ifft(u0_hat)
        else: raise ValueError("u0 or u0_hat must be given.")

        self.N  = len(self.u0)
        self.P  = P
        self.dx = P/self.N

        self.k = nf.fftfreq(self.N, d=self.dx) * 2*np.pi
        self.mk = self.dispersion(self.k)
        self.ddx = 1j*self.k # the derivative (d/dx)

        self.dt = dt
        self.T = T
        self.save_every = save_every

    @abstractmethod
    def dispersion(self,k):
        """Return the Fourier symbol m(k) as an array"""

    def nonlinear_rhs(self, u_hat_in):
        nl_hat = nf.fft(nf.ifft(u_hat_in)**2)
        return -0.5*self.ddx*nl_hat

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

        E = np.exp(-self.ddx*self.mk*self.dt / 2)
        
        while t_current < self.T - 1e-14:
            h = min(self.dt , self.T- t_current) #Truncating final step
            if h != self.dt:
                E = np.exp(-self.ddx*self.mk*h/ 2)

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
    """Solving the KdV equation over time (0,T)"""
    def dispersion(self, k):
        return 1 - k**2 /6

class Whitham(Solver):
    """Solving the Withams equation over time (0,T)"""
    def dispersion(self, k):
        mk = np.ones_like(k)
        nz = k != 0
        mk[nz] = np.sqrt( np.tanh(k[nz]) / k[nz])
        return mk

class fKdV(Solver):
    """Solving the fKdV equation over time (0,T)"""
    def __init__(self, P, dt, T, alpha,u0 = None, u0_hat = None, save_every:int = 1):
            self.alpha = alpha
            super().__init__( P = P,dt=dt, T=T, u0=u0, u0_hat=u0_hat, save_every=save_every)
        
    def dispersion(self, k):
        if self.alpha < 0:
            mk = np.zeros_like(k)
            nz = k != 0
            mk[nz] = - np.abs(k[nz])**self.alpha
            return mk     
        return - np.abs(k)**self.alpha


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

    kdv = KdV(u0=u0, P=P,dt= 5e-4, T=1)
    t, u = kdv.splitStep()

    anim = animate_solution(x,u)
    plt.show()