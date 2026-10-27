import numpy as np 
import numpy.fft as nf 
import matplotlib.pyplot as plt 
from solvers.help import *
from camera.camera_file import Camera

from abc import ABC, abstractmethod

class SpectralNewton(ABC):
    def __init__(self, P, target_amp, mean , camera:Camera):
        self.P          = P 
        self.target_amp        = target_amp 
        self.mean       = mean
        self.N          = round(P/camera.dx)
        assert abs(self.N * camera.dx - P) < 1e-9, "P/dx must be an integer"
        self.k          = 2*np.pi*nf.rfftfreq(self.N,d=camera.dx)
                
        
        self.mk = self.dispersion()
        self.M = len(self.k) - 1
        self.hw = self._height_weights()
        
    @abstractmethod
    def dispersion(self):
        """Return the Fourier symbol m(k) as an array"""
        
    def build_phi_hat(self, u):
        phi_hat = u[:-2]
        c       = u[-2]
        B       = u[-1]
        
        return phi_hat, c, B
    
    def get_physical(self, phi_hat):
        return nf.irfft(self.N*phi_hat, n = self.N)

    def _height_weights(self):
        w = np.zeros(self.M +1)
        w[1::2] = 2.0
        if self.N%2 == 0 and self.M % 2 == 1: # Nyquist index M is odd: weight should be 1
            w[-1] = 1.0
        return w

    def product_hat(self, a_hat, b_hat):
        """3/2-rule, coefficients of a*b"""
        N,M = self.N, self.M
        Np = 3*N//2

        a = np.zeros(Np//2 +1, dtype=complex)
        b = np.zeros(Np//2 +1, dtype=complex)

        a[:M] = a_hat[:M]
        b[:M] = b_hat[:M]

        a_fine = nf.irfft(Np*a, n=Np)
        b_fine = nf.irfft(Np*b, n=Np)
        p_hat = nf.rfft(a_fine*b_fine)/Np

        out = np.zeros(M+1, dtype=complex)
        out[:M] = p_hat[:M]
        return out
    
    def nonlinear_hat(self, phi_hat):
        return 0.5*self.product_hat(phi_hat,phi_hat)
    
    def residual(self, u, amp):
        phi_hat, c, B = self.build_phi_hat(u)
        nl_hat = self.nonlinear_hat(phi_hat)
        
        F = (self.mk - c)*phi_hat + nl_hat
        F[0] -= B
        
        mean_eq = phi_hat[0] - self.mean
        amp_eq  = np.sum(self.hw.copy()*phi_hat) - amp
        
        return np.concatenate([
            np.real_if_close(F),
            [mean_eq],
            [amp_eq]
        ])
        
    def jv(self, v_hat, phi_hat, c):
        return (self.mk-c)*v_hat + self.product_hat(phi_hat, v_hat)
    
    def jacobian(self, u):
        phi_hat, c, B = self.build_phi_hat(u)
        J = np.zeros((self.M+3, self.M+3))
        
        for j in range(self.M+1):
            e = np.zeros(self.M +1)
            e[j] = 1.0
            
            J[:self.M+1,j] = np.real(self.jv(e,phi_hat,c))
        J[:self.M+1,-2] = -np.real(phi_hat) #dF/dc
        J[0,-1]         = -1          #dF/dB
        
        J[self.M+1,0] = 1 # For the mean value
        J[self.M+2,:self.M+1] = self.hw # for the amplitude 
        
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
            u[1] = amp/2 #updating target amp
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
    def dispersion(self):
        return kdVDispersion(self.k)
    
class WhithamNewton(SpectralNewton):
    """Newton solver for Whithams k"""
    def dispersion(self):
        return whithamDispersion(self.k)

class fKdVNewton(SpectralNewton):
    """Newton solver for fKdV: $m(k) = -|k|^a"""
    def __init__(self, P, target_amp, mean, camera, alpha):
        self.alpha = alpha
        super().__init__(P, target_amp, mean, camera)
        
    def dispersion(self):
        return fKdVDispersion(self.alpha)(self.k)

class SolitonNewton(SpectralNewton):
    def _weights(self):
        w = np.full(self.M + 1, 2.0); w[0] = 1.0; w[-1] = 1.0
        return w                                  # phi(0) = sum(w * phi_hat)

    def residual(self, u, amp):
        phi_hat, c, B = self.build_phi_hat(u)
        F = (self.mk - c)*phi_hat + self.nonlinear_hat(phi_hat)
        F[0] -= B
        peak_eq = np.sum(self._weights()*phi_hat) - amp
        return np.concatenate([np.real_if_close(F), [B], [peak_eq]])

    def jacobian(self, u):
        J = super().jacobian(u)                   # rows 0..M are unchanged
        J[self.M+1, :] = 0; J[self.M+1, -1] = 1   # d(B)/dB
        J[self.M+2, :] = 0; J[self.M+2, :self.M+1] = self._weights()
        return J

    def solve_soliton(self, amp):
        # exact KdV guess for 1 - k^2/6 (see below); Newton polishes it
        kappa = np.sqrt(amp/2)
        x = self.dx*np.arange(self.N)
        d = (x + self.P/2) % self.P - self.P/2    # distance to nearest crest
        phi0 = nf.rfft(amp/np.cosh(kappa*d)**2)/self.N
        u0 = np.concatenate([phi0.real, [1 + amp/3, 0.0]])
        self.u = self.newton(u0, amp)
        self.phi_hat, self.c, self.B = self.u[:-2], self.u[-2], self.u[-1]

class KdVSoliton(SolitonNewton):
    def dispersion(self):
        return kdVDispersion(self.k)
    
    
def plot(u_list,names, N):
    fig, axes = plt.subplots(1,2)
    for sol,name in zip(u_list,names):
        phi_hat = sol[:-2]
        # c_sol , B_sol = sol[-2], sol[-1]
        phi_sol = nf.irfft(N*phi_hat, n = N)
        
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

    N = 2**6
    amp = 0.05*2
    mean = 0.0
    P = 2*2*np.pi
    
    c1 = Camera(P,N,1)

    kdv = KdVNewton(P=P, target_amp= amp, mean= mean , camera=c1)
    kdv.solve_w_continuation(n_steps=10)
    # kdv._plot_()
    
    whitham = WhithamNewton(P, amp, mean, camera=c1)
    whitham.solve_w_continuation()

    fkdv = fKdVNewton(P,amp, mean, camera=c1, alpha= 2) # Same as KdV
    fkdv.solve_w_continuation()
    # # fkdv._plot_()
    names = ['KdV' , 'Whitham' , fr'fKdV $\alpha = $ {fkdv.alpha}']
    u_list = [kdv.u, whitham.u , fkdv.u]

    plot(u_list,names, N=N)

    # sol = KdVSoliton(20*np.pi,amp, mean, camera=c1)
    # sol.solve_w_continuation()
    # sol._plot_()