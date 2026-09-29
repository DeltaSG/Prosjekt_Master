# Class for the spectral solver, look at plan point 5.1 
# HSG 28.09
import numpy as np 
import numpy.fft as nf
import matplotlib.pyplot as plt 

from abc import ABC, abstractmethod



class SpectralSolver(ABC):
    def __init__(self,P, target_amp, mean, N):
        self.P          = P
        self.N          = N
        self.dx         = P/N 

        self.k = 2*np.pi*nf.rfftfreq(self.N, self.dx)
        self.mask = (np.abs(self.k) < (2/3)*np.max(np.abs(self.k)))

        self.target_amp = target_amp
        self.mean       = mean


        self.mk         = self.diespersion(self.k)
        self.M = len(self.mk) - 1

    @abstractmethod
    def diespersion(self, k):
        """Return the Fourier symbol m(k) as an array"""

    def build_phi_hat(self, u):
        phi_hat = u[:-2]
        c = u[-2]
        B = u[-1]
        return phi_hat, c, B

    def get_physical_phi(self, phi_hat):
        return nf.irfft(phi_hat, n=self.N)

    def nonlinear_hat(self, phi_hat):
        phi = self.get_physical_phi(phi_hat*self.mask)
        nl = 0.5*phi**2
        return nf.rfft(nl)
    
    def residual(self, u, amp):
        phi_hat, c, B = self.build_phi_hat(u)
        nl_hat = self.nonlinear_hat(phi_hat)

        F = (self.mk-c)*phi_hat + nl_hat
        F[0] -= B*self.N

        mean_eq = phi_hat[0] - self.N*self.mean
        amp_eq  = phi_hat[1] - self.N*amp/2

        return np.concatenate([
            np.real_if_close(F),
            [mean_eq],
            [amp_eq]
        ])

    def jv(self, v_hat, phi_hat, c):
        phi = self.get_physical_phi(phi_hat*self.mask)
        v   = self.get_physical_phi(v_hat)
        return(
            (self.mk - c)*v_hat + nf.rfft(phi*v)
        )

    def jacobian(self, u):
        phi_hat, c, B = self.build_phi_hat(u)

        J = np.zeros((self.M+3, self.M+3))

        for j in range(self.M +1):
            e = np.zeros(self.M +1)
            e[j] = 1.0

            J[:self.M+1,j] = np.real( self.jv(e,phi_hat, c))

        J[:self.M+1, -2] = -np.real(phi_hat)    #dF/dc
        J[0,-1]          = -self.N              #dF/dB
        #amp and mean constraints

        J[self.M+1,0] = 1
        J[self.M+2,1] = 1

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

    def solve_w_continuation(self, n_steps= 50):
        u = np.zeros(self.M+3)
        u[-2] = 1 - (2*np.pi/self.P)**2 / 6 # inital guess, good for amp -> 0

        for amp in np.linspace(self.target_amp/n_steps, self.target_amp, n_steps):
            u[1] = self.N*amp/2 #updating target amp
            u = self.newton(u, amp)

        self.u = u
        self.phi_hat = self.u[:-2]
        self.c       = self.u[-2]
        self.B       = self.u[-1]

    def _plot_(self):
        phi_hat = self.u[:-2]
        c_sol , B_sol = self.u[-2], self.u[-1]

        phi_sol = self.get_physical_phi(phi_hat)

        print(f"amplitude = {(np.max(phi_sol) - np.min(phi_sol))/2}")
        print(f"Wave speed c = {c_sol}")
        print(f"Constant B = {B_sol}")

        fig, axes = plt.subplots(1,2)
        axes[0].plot(phi_sol, label = r"$\phi$")
        axes[1].plot(self.k, phi_hat, label = r"$\hat{\phi}$")

        axes[0].grid()
        axes[0].legend()
        axes[1].grid()
        axes[1].legend()
        plt.show()


class KdVSolver(SpectralSolver):
    """Newton solver for KdV"""
    def diespersion(self, k):
        return 1 - k**2 /6

class WhithamSolver(SpectralSolver):
    """Newton solver for Whithams k"""
    def diespersion(self, k):
        mk = np.ones_like(k)
        nz = k != 0
        mk[nz] = np.sqrt( np.tanh(k[nz]) / k[nz])
        return mk

class fKdVSolver(SpectralSolver):
    """Newton solver for fKdV"""
    def __init__(self, P, target_amp, mean, N, alpha):
         self.alpha = alpha
         super().__init__(P, target_amp, mean, N)
        
    def diespersion(self, k):
        if self.alpha < 0:
            mk = np.zeros_like(k)
            nz = k != 0
            mk[nz] = - np.abs(k[nz])**self.alpha
            return mk     
        return - np.abs(k)**self.alpha



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
    N = 2**6
    amp = 0.05
    mean = 0.0
    P = 2*np.pi

    kdv = KdVSolver(P=P, target_amp= amp, mean= mean , N = N)
    kdv.solve_w_continuation(n_steps=10)
    # wkdv._plot_()

    whitham = WhithamSolver(P, amp, mean, N)
    whitham.solve_w_continuation()
    # witham._plot_()

    fkdv = fKdVSolver(P,amp, mean, N, alpha= 2) # Same as KdV
    fkdv.solve_w_continuation()
    # fkdv._plot_()

    names = ['KdV' , 'Whitham' , r'fKdV $\alpha = 1$']
    u_list = [kdv.u, whitham.u , fkdv.u]

    plot(u_list,names, N=N)


