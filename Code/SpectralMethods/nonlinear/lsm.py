# A class that takes in a tensor of data and uses the least squares method to estimate the symbols
# To take in comlex waves, and non-periodic data we are using FDM to estimate the time derivative
# HSG 05.10.26


import numpy as np
import numpy.fft as nf
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from linear.help import *
import pandas as pd

class LSM():
    """A class that takes in a tensor of data and uses the least squares method to estimate the symbols"""
    def __init__(self, eta_list, dt):
        self.eta_list = eta_list
        self.dt = dt

    # @abstractmethod
    # def symbol(self,k):
    #     """Override with the methods symbols l and n"""

    def lsm(self):
        """No dealiasing"""
        eta_hat_list = []
        nl_hat_list  = []
        dHdt_hat_list= []

        for eta in self.eta_list:
            eta_hat = nf.fft(eta, axis=1)
            # add (current) highest non-zero k test
            nl_hat = nf.fft(eta**2, axis=1)
            dHdt_hat= nf.fft(np.gradient(eta, self.dt, axis=0, edge_order=2), axis=1)

            eta_hat_list.append(eta_hat)
            nl_hat_list.append(nl_hat)
            dHdt_hat_list.append(dHdt_hat)

        
        H = np.stack(eta_hat_list, axis=0)
        H2 = np.stack(nl_hat_list, axis=0)
        dHdt = np.stack(dHdt_hat_list, axis=0)

        def to_freq_first(arr):
            """Move frequency to axis 0, and flatten (soltion, time) into one sample axis"""
            arr = np.moveaxis(arr, -1,0)
            return arr.reshape(arr.shape[0], -1)
        # crop lists in k, we only want the modes that are significantly non-zero
        Hr      = to_freq_first(H)
        H2r     = to_freq_first(H2)
        dHdtr   = to_freq_first(dHdt)

        A = np.stack(
            [Hr, H2r] , axis= 1
        )

        A_pinv = np.linalg.pinv(A)
        X = np.einsum('fs,fsc->fc', dHdtr, A_pinv)

        self.ell    = X[:,0]
        self.n      = X[:,1]

        return X


    def fit_symbol(self, k, idx_range = None, deg = 3):
        """Fit the symbols to a polynomial in k"""
        order = np.argsort(k)
        k = k[order]
        ell = self.ell[order]
        n = self.n[order]
        if idx_range is not None:
            k = k[idx_range[0]:idx_range[1]]
            ell = ell[idx_range[0]:idx_range[1]]
            n = n[idx_range[0]:idx_range[1]]

        self.ell_fit = np.polyfit(k, ell, deg=deg)
        self.n_fit   = np.polyfit(k, n, deg=deg)

        return self.ell_fit, self.n_fit 


    def plot_symbol_fit(self, k, idx_range = None, ell_t = lambda k: k**3/6 - k, n_t = lambda k: -0.5*k):
        order = np.argsort(k)
        k = k[order]
        ell = self.ell[order]
        n = self.n[order]
        if idx_range is not None:
            k = k[idx_range[0]:idx_range[1]]
            ell = ell[idx_range[0]:idx_range[1]]
            n = n[idx_range[0]:idx_range[1]]

        ell_theory = ell_t(k)
        n_theory   = n_t(k)

        fig, ax = plt.subplots(1,2)
        ax[0].plot(k, ell.imag, label = "ell", marker='o', linestyle='-', markersize=3)
        ax[0].plot(k, ell_theory, label = "theory", marker='x', linestyle=':', markersize=3)
        ax[0].legend()
        ax[0].grid(True)
    
        ax[1].plot(k, n.imag, label = "n", marker='o', linestyle='-', markersize=3)
        ax[1].plot(k, n_theory, label = "theory", marker='x', linestyle=':', markersize=3)
        ax[1].legend()
        ax[1].grid(True)
    
        plt.show()

        
if __name__ == "__main__":
    from nonlinear.spectralSolvers import KdVSolver
    from nonlinear.solvers import KdV


    w1 = KdVSolver(P = 2*np.pi, target_amp = 0.1, N = 2**8, mean = 0.0)
    w1.solve_w_continuation()
    w2 = KdVSolver(P = 2*np.pi, target_amp = 0.25, N = 2**8, mean = 0.0)
    w2.solve_w_continuation()

    # kdv1 = KdV(P = 2*np.pi, dt = 5e-3, T = 1, u0 = np.fft.irfft(w1.u[:-2] + w2.u[:-2], n = 2**8), save_every=5)
    # kdv1.splitStep()

    kdv1 = KdV(P = 2*np.pi, dt = 5e-3, T = 1, u0 = np.fft.irfft(w1.u[:-2], n = 2**8), save_every=1)
    kdv1.splitStep()
    kdv2 = KdV(P = 2*np.pi, dt = 5e-3, T = 1, u0 = np.fft.irfft(w2.u[:-2], n = 2**8), save_every=1)
    kdv2.splitStep()

    lsm = LSM([kdv1.u, kdv2.u], kdv1.dt)
    # lsm = LSM([kdv1.u], kdv1.dt*kdv1.save_every)
    X = lsm.lsm()

    k = np.fft.fftfreq(kdv1.N, d = kdv1.P/kdv1.N)*2*np.pi
    idx = kdv1.N//2 - 1
    idx_range = (kdv1.N//2 - idx +1, kdv1.N//2 + idx)

    ell , n= lsm.fit_symbol(k, idx_range = idx_range, deg = 5)
    print(fit_table(ell, n, part="imag"))


    order = np.argsort(k)
    half = kdv1.N//2
    k_o = k[order]
    X_0 = X[order,0]
    X_1 = X[order,1]


    ii = idx
    idx = half + ii
    idx_neg = half - ii + 1
    
    lsm.plot_symbol_fit(k, idx_range = (idx_neg, idx), ell_t = lambda k: k**3/6 - k, n_t = lambda k: -0.5*k)