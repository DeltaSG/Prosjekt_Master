# This file shall contain the LSM for a general problem


import numpy as np
import numpy.fft as nf
import pandas as pd
import matplotlib.pyplot as plt
from solvers.help import *
from camera.camera_file import Camera

class LSM():
    """A class that takes in a tensor of data and uses the least squares method to estimate the symbols"""
    def __init__(self, eta_list, camera:Camera, drop_frames:int = 1):
        self.eta_list = eta_list
        self.camera = camera
        self.drop_frames = drop_frames
        self.ddx = 1j*camera.k

    def lsm(self):
        """No dealiasing"""
        eta_hat_list = []
        nl_hat_list  = []
        dHdt_hat_list= []

        sl = slice(self.drop_frames, -self.drop_frames)

        for eta in self.eta_list:
            M = len(eta[0])
            assert self.camera.N % M == 0, f" {N = } is not divisible by {M = }"
            m = self.camera.N // M
            if m != 1:
                # Adding copies of the wave so that it covers L
                eta = np.tile(eta, m)
            eta_hat = -self.ddx * nf.fft(eta, axis=1)
            eta_hat_full = nf.fft(eta, axis=1)
            nl_hat = - self.ddx * dealiased_square_hat(eta_hat_full)
            dHdt_hat= nf.fft(np.gradient(eta, self.camera.dt, axis=0, edge_order=2), axis=1)
            # time derivative in Fourier space:
            # dt_f = 
            #removing edge frames
            
            eta_hat_list.append(eta_hat[sl])
            nl_hat_list.append(nl_hat[sl])
            dHdt_hat_list.append(dHdt_hat[sl])

        
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
        order = np.argsort(self.camera.k)
        k = self.camera.k[order]
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
        if idx_range is None:
            idx_range = (2 , -1)
        order = np.argsort(self.camera.k)
        k = self.camera.k[order]
        ell = self.ell[order]
        n = self.n[order]
        k = k[idx_range[0]:idx_range[1]]
        ell = ell[idx_range[0]:idx_range[1]]
        n = n[idx_range[0]:idx_range[1]]

        ell_theory = ell_t(k)
        n_theory   = n_t(k)

        fig, ax = plt.subplots(1,2, figsize = (10,3))
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
    from camera.spectralCamera import KdVNewton
    from camera.pdeCamera import KdV
    N = 2**6
    L = 2*np.pi
    dt = 1e-3
    c1 = Camera(L,N,dt)

    w1 = KdVNewton(P = 4*np.pi, target_amp = 0.1, mean = 0.0, camera  =c1)
    w1.solve_w_continuation()
    w2 = KdVNewton(P = 2*np.pi, target_amp = 0.25, camera  =c1, mean = 0.0)
    w2.solve_w_continuation()

    # kdv1 = KdV(P = 2*np.pi, dt = 5e-3, T = 1, u0 = np.fft.irfft(w1.u[:-2] + w2.u[:-2], n = 2**8), save_every=5)
    # kdv1.splitStep()

    kdv1 = KdV(camera  =c1, T = 1, u0 = np.fft.irfft(w1.u[:-2], n = N), save_every=1)
    kdv1.splitStep()
    kdv2 = KdV(camera  =c1, T = 1, u0 = np.fft.irfft(w2.u[:-2], n = N), save_every=1)
    kdv2.splitStep()

    lsm = LSM([kdv1.u, kdv2.u], camera  =c1)
    # lsm = LSM([kdv1.u], kdv1.dt*kdv1.save_every)
    X = lsm.lsm()

    idx = kdv1.N//2 - 1
    idx_range = (kdv1.N//2 - idx, kdv1.N//2 + idx + 1)

    ell , n= lsm.fit_symbol(idx_range = idx_range, deg = 5)
    print(fit_table(ell, n, part="real"))


    order = np.argsort(c1.k)
    half = kdv1.N//2
    k_o = c1.k[order]
    X_0 = X[order,0]
    X_1 = X[order,1]


    ii = idx
    idx = half + ii
    idx_neg = half - ii + 1
    
    lsm.plot_symbol_fit(idx_range = (idx_neg, idx), ell_t = lambda k: 1-k**2/6)