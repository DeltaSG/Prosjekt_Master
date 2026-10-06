# Code following the structure of ikdv and the note from Mats, Douglas, mm. 
# HSG 29.09

from traceback import print_list

import numpy as np 
import numpy.fft as nf
import matplotlib.pyplot as plt 
from abc import ABC, abstractmethod

from linear.help import *
from nonlinear.solvers import *
from nonlinear.spectralSolvers import * 

class WaveCollection(ABC):
    """Collection of different waves for LSM"""
    def __init__(self, P, N, dt, T, amp_list, mean_list):
        self.P  = P
        self.N  = N
        self.dx = P/N
        self.T  = T
        self.dt = dt
        self.w  = 2*np.pi*nf.fftfreq(int(T/dt)+1, dt)

        self.k = 2*np.pi*nf.fftfreq(N, self.dx)

        self.amp_list = amp_list
        self.mean_list = mean_list

    @abstractmethod
    def wave_function(self, target_amp, mean):
        """Override with the travelling wave solver"""

    @abstractmethod
    def wave_solver(self, u0):
        """Override with the PDE solver"""

    @abstractmethod
    def symbol(self,k):
        """Override with the methods symbols l and n"""

    def make_travelling_waves(self):
        H0 = []

        for amp, mean in zip(self.amp_list, self.mean_list):
            eta = self.wave_function(target_amp=amp, mean=mean)
            eta.solve_w_continuation()
            H0.append(nf.irfft(eta.phi_hat, n=self.N))
        self.H0 = H0

    def solve_forward(self):
        eta_list = []
        for eta in self.H0:
            kdvs = self.wave_solver(eta)
            t,u  = kdvs.splitStep()
            eta_list.append(u) 
        self.eta_list = eta_list


    def LSM(self):
        eta_hat_list = []
        nl_hat_list  = []
        dHdt_hat_list= []

        for eta in self.eta_list:
            eta_hat = nf.fft(eta, axis=1) #fft_x
            # eta_pad = pad(eta, self.N)

            mask = (np.abs(self.k) < (2/3)*np.max(np.abs(self.k)))
            eta_m = nf.ifft(eta_hat*mask, axis=1)
            eta2_hat = nf.fft(eta_m**2, axis=1)
            # #Dealiasing by padding
            # Npad = 3*self.N//2
            # eta2_hat_pad =  nf.fft(eta_pad**2, axis= 1) * 2/3
            # eta2_hat = np.zeros((int((self.T / self.dt) +1) , self.N), dtype=complex)
            # N2 = self.N//2
            # eta2_hat[:,:N2] = eta2_hat_pad[:,:N2]
            # eta2_hat[:,-(N2-1):] = eta2_hat_pad[:,-(N2-1):]

            # dEtadt  = nf.ifft(
            #     1j*self.w[:,None] * nf.fft(eta_hat, axis=0), axis=0
            # ) # fft_t

            dEtadt = nf.fft(np.gradient(eta, self.dt, axis = 0, edge_order=2), axis=1)

            eta_hat_list.append(eta_hat)
            nl_hat_list.append(eta2_hat)
            dHdt_hat_list.append(dEtadt)
        
        self.H =     np.stack(eta_hat_list,  axis=0)
        H2=     np.stack(nl_hat_list,   axis=0)
        dHdt =  np.stack(dHdt_hat_list, axis=0)

        def to_freq_first(arr):
            """Move frequency to axis 0, and flatten (soltion, time) into one sample axis"""
            arr = np.moveaxis(arr, -1,0)
            return arr.reshape(arr.shape[0], -1)

        Hr      = to_freq_first(self.H)
        H2r     = to_freq_first(H2)
        dHdtr   = to_freq_first(dHdt)

        sgn = -1
        ik = (sgn*1j*self.k)[:,None]

        A = np.stack(
            [ik*Hr, ik*H2r] , axis= 1
        )

        A_pinv = np.linalg.pinv(A)
        X = np.einsum('fs,fsc->fc', dHdtr, A_pinv)


        i0 = np.argmin(np.abs(self.k))
        i1, i2 = i0+1, i0+2
        
        for arr in (X[:,0], X[:,1]):
            arr[i0] = (4*arr[i1] - arr[i2]) / 3

        self.ell    = X[:,0]
        self.n      = X[:,1]

        return X

    def solve(self):
        self.make_travelling_waves()
        self.solve_forward()
        X = self.LSM()
        return X


    def _plot_(self, idx = None):
        fig, axes = plt.subplots(2,2,figsize = (10,4))
        if idx is None:
            idx = int(3*self.P/(4*np.pi) +1)
        a_neg = self.N//2 - idx
        a_pos = self.N//2 + idx +1

        order = np.argsort(self.k)
        k = self.k[order]

        ell_theory = self.symbol(k)
        n_theory   = np.ones_like(k) / 2

        ell = self.ell[order]
        # ell[k==0] = np.nan
        n = self.n[order]
        # n[k==0] = np.nan

        H_sorted = self.H[:,0,order]
    
        axes[0,0].plot(k[a_neg:a_pos], ell[a_neg:a_pos].real, '.', label=r'estimated $\mathbb{Re}(\ell)$')
        axes[0,0].plot(k[a_neg:a_pos], ell_theory[a_neg:a_pos].real, '-', label='theory')
    
        axes[0,0].set_xlabel(r'$\xi$')
        axes[0,0].set_ylabel(r'Re($\ell$)')
        axes[0,0].set_title(r'$\ell(\xi)$')
        axes[0,0].legend()
    
        axes[0,1].plot(k[a_neg:a_pos], n[a_neg:a_pos].real, '.', label='estimated Re($n$)')
        axes[0,1].plot(k[a_neg:a_pos], n_theory[a_neg:a_pos].real, '-', label=r'theory: $0.5$')
        axes[0,1].set_xlabel(r'$\xi$')
        axes[0,1].set_ylabel(r'Re($n$)')
        axes[0,1].set_title(r'$n(\xi)$')
        axes[0,1].legend()
        
        
        axes[0,0].grid()
        axes[0,1].grid()

        for i,amp in enumerate(self.amp_list):
            axes[1,0].semilogy(k[self.N//2:a_pos], H_sorted[i, self.N//2:a_pos].real, label=f'amp={amp}')
            # axes[1,1].plot(k[a_neg:a_pos], H_sorted[i, a_neg:a_pos].imag, label=f'amp={amp}')
        axes[1,0].set_xlabel(r'$\xi$')
        axes[1,0].set_ylabel(r'Re$\widehat{\eta}$')
        # axes[1,1].set_ylabel(r'Im$\widehat{\eta}$')
        # axes[1,0].set_title(r'$H(\xi)$')
        axes[1,0].legend()
        axes[1,0].grid()
        # axes[1,1].grid()

        plt.tight_layout()
        plt.show()

    def fit_symbol(self, idx_range = None, deg = 3):
        """Fit the symbols to a polynomial in k"""
        order = np.argsort(self.k)
        k = self.k[order]
        ell = self.ell[order]
        n = self.n[order]
        if idx_range is not None:
            k = k[idx_range[0]:idx_range[1]]
            ell = ell[idx_range[0]:idx_range[1]]
            n = n[idx_range[0]:idx_range[1]]

        self.ell_fit = np.polyfit(k, ell, deg=deg)
        self.n_fit   = np.polyfit(k, n, deg=deg)

        return self.ell_fit, self.n_fit
 
class KdVLSM(WaveCollection):
    def wave_function(self, target_amp, mean):
        return KdVSolver(P=self.P, target_amp= target_amp, mean=mean, N = self.N)
    def wave_solver(self, u0):
        return KdV(self.P, self.dt, self.T,u0=u0)

    def symbol(self,k):
        return 1-k**2 / 6

class fKdVLSM(WaveCollection):
    def __init__(self, P, N, dt, T, amp_list, mean_list, alpha):
        self.alpha = alpha
        super().__init__(P, N, dt, T, amp_list, mean_list)

    def wave_function(self, target_amp, mean):
        return fKdVSolver(P = self.P, target_amp=target_amp, mean=mean, N = self.N , alpha=self.alpha)
    def wave_solver(self, u0):
        return fKdV(P=self.P, dt = self.dt, T=self.T, alpha=self.alpha, u0=u0)

    def symbol(self,k):
        if self.alpha < 0:
            mk = np.zeros_like(k)
            nz = k != 0
            mk[nz] = - np.abs(k[nz])**self.alpha
            return mk     
        return - np.abs(k)**self.alpha
        

class WhithamLSM(WaveCollection):
    def wave_function(self, target_amp, mean):
        return WhithamSolver(P=self.P, target_amp= target_amp, mean=mean, N = self.N)
    def wave_solver(self, u0):
        return Whitham(self.P, self.dt, self.T,u0=u0)

    def symbol(self,k):
        mk = np.ones_like(k)
        nz = k != 0
        mk[nz] = np.sqrt( np.tanh(k[nz]) / k[nz])
        return mk
    
if __name__ == '__main__':
    amp_l   = [0.05,    0.075,  0.1,    0.125,  0.15]
    mean_l  = [0.0,     0.0,    0.0,    0.0,    0.0]

    P = 2*np.pi
    N = 2**6
    dt = 5e-3
    T= 1

    # wp = WhithamLSM(P=P,N=N, dt=dt, T=T, amp_list=amp_l, mean_list=mean_l)
    # wp = fKdVLSM(P=P,N=N, dt=dt, T=T, amp_list=amp_l, mean_list=mean_l, alpha=2)
    wp = KdVLSM(P=P,N=N, dt=dt, T=T, amp_list=amp_l, mean_list=mean_l)

    X = wp.solve()
    idx = 10
    wp.fit_symbol(idx_range = (N//2-idx+1, N//2+idx), deg = 5)


    print(fit_table(wp.ell_fit, wp.n_fit, part="real"))
    wp._plot_(idx)
    # for t in range(0,int(T/dt), 40):
    #     for i, amp in enumerate(amp_l):

    #         plt.semilogy(wp.k[:N//2-1], wp.H[i,t,:N//2-1].real, label=f'amp={amp}')
    #     plt.title(f't={t*dt:.2f}')
    #     plt.xlabel(r'$\xi$'); plt.ylabel(r'$|\hat H(\xi)|$')
    #     plt.legend()
    #     plt.show()