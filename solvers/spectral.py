# This file shall contain the Newton solvers for the waves

import numpy as np 
import numpy.fft as nf 
import matplotlib.pyplot as plt 
from solvers.help import *

from abc import ABC, abstractmethod # For making abstract classes / methods 


class SpectralNewton(ABC):
    def __init__(self, P, target_amp, mean, dx):
        self.P          = P
        self.dx          = dx
        self.N         = round(P/dx)
        assert abs(self.N*dx - P) < 1e-9, "P/dx must be an integer"
        self.k          = 2*np.pi*nf.rfftfreq(self.N,d=self.dx)
        
        self.target_amp = target_amp
        self.mean       = mean
        
        self.mk = self.dispersion(self.k)
        self.M = len(self.k) - 1
        
    @abstractmethod
    def dispersion(self, k):
        """Return the Fourier symbol m(k) as an array"""
        
    def build_phi_hat(self, u):
        phi_hat = u[:-2]
        c       = u[-2]
        B       = u[-1]
        
        return phi_hat, c, B
    
    def get_physical(self, phi_hat):
        return nf.irfft(phi_hat, n = self.N)
    
    def nonlinear_hat(self, phi_hat):
        phi = self.get_physical(phi_hat)
        return 0.5* nf.rfft(phi**2)
    
    def residual(self, u, amp):
        phi_hat, c, B = self.build_phi_hat(u)
        nl_hat = self.nonlinear_hat(phi_hat)
        
        F = (self.mk - c)*phi_hat + nl_hat
        F[0] -= B*self.N
        
        mean_eq = phi_hat[0] - self.N*self.mean
        amp_eq  = phi_hat[1] - self.N*amp/2
        
        return np.concatenate([
            np.real_if_close(F),
            [mean_eq],
            [amp_eq]
        ])
        
    def jv(self, v_hat, phi_hat, c):
        phi = self.get_physical(phi_hat)
        v   = self.get_physical(v_hat)
        return (self.mk-c)*v_hat + nf.rfft(phi*v)
    
    def jacobian(self, u):
        phi_hat, c, B = self.build_phi_hat(u)
        J = np.zeros((self.M+3, self.M+3))
        
        for j in range(self.M+1):
            e = np.zeros(self.M +1)
            e[j] = 1.0
            
            J[:self.M+1,j] = np.real(self.jv(e,phi_hat,c))
        J[:self.M+1,-2] = -np.real(phi_hat) #dF/dc
        J[0,-1]         = -self.N           #dF/dB
        
        J[self.M+1,0] = 1 # For the mean value
        J[self.M+2,1] = 1 # for the amplitude 
        
        return J
    
    def linesearch(self, u, amp, err, delta):
        alpha = 1

        while alpha > 1e-8:
            u_trial = u + alpha*delta

            if np.linalg.norm(
                self.residual(u_trial, amp)
            ) < err: break

            alpha *= 0.5
        return u_trial
    
    def newton(self, u0, amp, tol = 1e-10, maxit = 50):
        u = u0.copy()

        for _ in range(maxit):
            F = self.residual(u, amp)
            err = np.linalg.norm(F)

            if err < tol: return u

            J = self.jacobian(u)
            delta = np.linalg.solve(J, -F)
            u = self.linesearch(u, amp, err, delta)
        raise RuntimeError('No convergence')

    def solve_w_continuation(self, n_steps= 50, u0 = None, amp0 = None):
        if u0 is not None:
            u = u0
        else: u = np.zeros(self.M+3)
        u[-2] = self.mk[1]

        if amp0 is not None:
            start = amp0
        else:  start = self.target_amp/n_steps
        
        for amp in np.linspace(start, self.target_amp, n_steps):
            u[1] = self.N*amp/2 #updating target amp
            u = self.newton(u, amp)

        self.u = u
        self.phi_hat = self.u[:-2]
        self.phi_hat_mid = self.phi_hat *(-1.0)**np.arange(len(self.phi_hat))
        self.c       = self.u[-2]
        self.B       = self.u[-1]
        
    def _plot_(self):
        phi_hat = self.u[:-2]
        c_sol , B_sol = self.u[-2], self.u[-1]

        phi_sol = self.get_physical(phi_hat)

        print(f"amplitude = {(np.max(phi_sol) - np.min(phi_sol))/2}")
        print(f"Wave speed c = {c_sol}")
        print(f"Constant B = {B_sol}")
        
        x = np.linspace(0,self.P, self.N, endpoint=False)

        fig, axes = plt.subplots(1,2)
        axes[0].plot(x,phi_sol, label = r"$\phi$")
        axes[1].plot(self.k, phi_hat, label = r"$\hat{\phi}$")

        axes[0].grid()
        axes[0].legend()
        axes[1].grid()
        axes[1].legend()
        plt.show()
        
class KdVNewton(SpectralNewton):
    """Newton solver for KdV"""
    def dispersion(self, k):
        return kdVDispersion(k)
    
class WhithamNewton(SpectralNewton):
    """Newton solver for Whithams k"""
    def dispersion(self, k):
        return whithamDispersion(k)

class fKdVNewton(SpectralNewton):
    """Newton solver for fKdV: $m(k) = -|k|^a"""
    def __init__(self, P, target_amp, mean, dx, alpha):
        self.alpha = alpha
        super().__init__(P, target_amp, mean, dx)
        
    def dispersion(self, k):
        return fKdVDispersion(self.alpha)(k)
    
    
def plot(u_list,names, N):
    fig, axes = plt.subplots(1,2)
    for sol,name in zip(u_list,names):
        phi_hat = sol[:-2]
        # c_sol , B_sol = sol[-2], sol[-1]
        phi_sol = nf.irfft(phi_hat, n = N)
        
        axes[0].plot(phi_sol, label = name, alpha = 0.7)
        axes[1].plot(phi_hat, label = name, alpha = 0.7)

    axes[0].set_title(r"$\phi$")
    axes[0].grid()
    axes[0].legend()
    axes[1].set_title(r"$\widehat{\phi}$")
    axes[1].grid()
    axes[1].legend()
    plt.show()

if __name__ == '__main__':

    def soliton_profile(x,c,x0=0.0):
        return (c/2)/np.cosh(np.sqrt(c)/2 * (x-x0))**2
    
    N = 2**6
    amp = 0.05*2
    mean = 0.0
    P = 2*2*np.pi
    dx = np.pi*1e-1

    #
    x = np.linspace(-P/2, P/2, N, endpoint=False)
    #


    k = 2*np.pi*nf.rfftfreq(n=N, d = P/N)

    kdv = KdVNewton(P=P, target_amp= amp, mean= mean , dx = dx)
    kdv.solve_w_continuation(n_steps=10)
    
    whitham = WhithamNewton(P, amp, mean, dx)
    whitham.solve_w_continuation()

    fkdv = fKdVNewton(P,amp, mean, dx, alpha= 2) # Same as KdV
    fkdv.solve_w_continuation()
    # fkdv._plot_()
    names = ['KdV' , 'Whitham' , fr'fKdV $\alpha = $ {fkdv.alpha}']
    u_list = [kdv.u, whitham.u , fkdv.u]

    plot(u_list,names, N=N)