# This file shall contain the LSM for a general problem


import numpy as np
import numpy.fft as nf
import pandas as pd
import matplotlib.pyplot as plt
from solvers.help import *

class LSM():
    """A class that takes in a tensor of data and uses the least squares method to estimate the symbols"""
    def __init__(self, eta_list, dt, P):
        self.eta_list = eta_list
        self.dt = dt
        
        # Assuming same dimensions on the dataset:
        N = len(eta_list[0][0])
        self.k = 2*np.pi*nf.fftfreq(N, d=P/N)
        self.ddx = 1j*self.k

    def lsm(self):
        """No dealiasing"""
        eta_hat_list = []
        nl_hat_list  = []
        dHdt_hat_list= []

        for eta in self.eta_list:
            eta_hat = -self.ddx * nf.fft(eta, axis=1)
            # add (current) highest non-zero k test
            nl_hat = - self.ddx * nf.fft(eta**2, axis=1)
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

        # ----- Looking at the two data columns -----------
        col_norm = np.linalg.norm(A, axis=2)
        A_n = A/np.where(col_norm[...,None]==0,1, col_norm[..., None])
        sv = np.linalg.svd(A_n, compute_uv=False)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.sv_ratio   = sv[:,1]/sv[:,0]
            self.cond       = sv[:,0]/sv[:,1]
        self.col_norm = col_norm
        # -------------------------------------------------

        A_pinv = np.linalg.pinv(A)
        X = np.einsum('fs,fsc->fc', dHdtr, A_pinv)

        self.ell    = X[:,0]
        self.n      = X[:,1]
        
        self.ell[0] = np.nan
        self.n[0] = np.nan

        return X


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

        m = np.isfinite(ell)
        self.ell_fit = np.polyfit(k[m], ell[m].real, deg)
        self.n_fit   = np.polyfit(k[m], n[m].real, deg)

        return self.ell_fit, self.n_fit 


    def plot_symbol_fit(self, idx_range = None, ell_t = lambda k:1- k**2/6 , n_t = lambda k: 0.5*np.ones_like(k)):
        order = np.argsort(self.k)
        k = self.k[order]
        ell = self.ell[order]
        n = self.n[order]
        if idx_range is not None:
            k = k[idx_range[0]:idx_range[1]]
            ell = ell[idx_range[0]:idx_range[1]]
            n = n[idx_range[0]:idx_range[1]]

        ell_theory = ell_t(k)
        n_theory   = n_t(k)

        fig, ax = plt.subplots(1,2, figsize = (15,3))
        ax[0].plot(k, ell.real, label = "ell", marker='o', linestyle='-', markersize=3)
        ax[0].plot(k, ell_theory, label = "theory", marker='x', linestyle=':', markersize=3)
        ax[0].legend()
        ax[0].grid(True)
    
        ax[1].plot(k, n.real, label = "n", marker='o', linestyle='-', markersize=3)
        ax[1].plot(k, n_theory, label = "theory", marker='x', linestyle=':', markersize=3)
        ax[1].legend()
        ax[1].grid(True)
    
        plt.show()

        
if __name__ == "__main__":
    from solvers.spectral import KdVNewton
    from solvers.pde import KdV


    w1 = KdVNewton(P = 2*np.pi, target_amp = 0.1, N = 2**8, mean = 0.0)
    w1.solve_w_continuation()
    w2 = KdVNewton(P = 2*np.pi, target_amp = 0.25, N = 2**8, mean = 0.0)
    w2.solve_w_continuation()

    # kdv1 = KdV(P = 2*np.pi, dt = 5e-3, T = 1, u0 = np.fft.irfft(w1.u[:-2] + w2.u[:-2], n = 2**8), save_every=5)
    # kdv1.splitStep()

    kdv1 = KdV(P = 2*np.pi, dt = 1e-3, T = 1, u0 = np.fft.irfft(w1.u[:-2], n = 2**8), save_every=1)
    kdv1.splitStep()
    kdv2 = KdV(P = 2*np.pi, dt = 1e-3, T = 1, u0 = np.fft.irfft(w2.u[:-2], n = 2**8), save_every=1)
    kdv2.splitStep()

    lsm = LSM([kdv1.u, kdv2.u], kdv1.dt, P=2*np.pi)
    # lsm = LSM([kdv1.u], kdv1.dt*kdv1.save_every)
    X = lsm.lsm()

    k = np.fft.fftfreq(kdv1.N, d = kdv1.P/kdv1.N)*2*np.pi
    id = 2
    idx = 5 #int(id*np.pi)#kdv1.N//2 - 1
    idx_range = (kdv1.N//2 - idx, kdv1.N//2 + idx + 1)

    ell , n= lsm.fit_symbol(idx_range = idx_range, deg = 5)
    print(fit_table(ell, n, part="real"))


    order = np.argsort(k)
    half = kdv1.N//2
    k_o = k[order]
    X_0 = X[order,0]
    X_1 = X[order,1]


    ii = idx
    idx = half + ii
    idx_neg = half - ii + 1
    
    lsm.plot_symbol_fit(idx_range = (idx_neg, idx), ell_t = lambda k: 1-k**2/6)