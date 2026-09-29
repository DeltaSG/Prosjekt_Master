# full inverse KdV time ind/dep
# HSG
# 24.09

import numpy as np 
import numpy.fft as nf
import matplotlib.pyplot as plt 
from KdV.soliton import Soliton
from linear.help import make_x_array
from linear.help import animate_solution

class Packet():
    """Collection of waves for LSM"""
    def __init__(self,L,N,T,M, c_list, x0_list):
        self.c_list  = c_list
        self.x0_list = x0_list
        
        self.L = L
        self.N = N
        self.T = T
        self.M = M
        self.x       = make_x_array(N, L, symmetric=True)
        
        self.k = 2*np.pi*np.fft.fftfreq(N, d=(self.x[1]-self.x[0]))
        
        return

        
    def make_solitons(self):
        phi_hat_list    = []
        nl_hat_list     = []
        dHdt_hat_list   = []
        
        for c,x0  in zip(self.c_list, self.x0_list):
            
            T_c = self.L / c
            t_c = make_x_array(self.M,T_c, symmetric=False)
            
            w = 2*np.pi*nf.fftfreq(self.M, d= t_c[1]-t_c[0])
            
            eta = Soliton(c,x0).wave(t_c, self.x)
            eta_hat  = nf.fft(eta,      axis=1)
            # H2 with 2/3 mask dealiasing
            k_max = np.max(np.abs(self.k))
            mask = (np.abs(self.k) < (2/3)*k_max)
            # mask = np.ones_like(eta)
            eta_m = nf.ifft(eta_hat*mask)
            eta2_hat = nf.fft((eta_m)**2,   axis=1)

            detadt_hat = nf.ifft(
                1j*w[:,None] * nf.fft(
                    eta_hat , axis = 0),
                axis = 0)
            phi_hat_list.append(eta_hat)
            nl_hat_list.append(eta2_hat)
            dHdt_hat_list.append(detadt_hat)
            
        self.H      = np.stack(phi_hat_list,    axis = 0)
        self.H2     = np.stack(nl_hat_list,     axis = 0)
        self.dHdt   = np.stack(dHdt_hat_list,   axis = 0)
        
        return 

    def LSM(self):
        
        def to_freq_first(arr):
            """Move frequency to axis 0,
            and flatten (soliton, time) into one sample axis"""
            arr = np.moveaxis(arr, -1,0)
            return arr.reshape(arr.shape[0], -1) 
        
        Hr = to_freq_first(self.H)
        H2r = to_freq_first(self.H2)
        dHdtr = to_freq_first(self.dHdt)
        
        A = np.stack(
            [Hr, H2r],
            axis =1
        )
        A_pinv = np.linalg.pinv(A)
        X = np.einsum('fs,fsc->fc', dHdtr, A_pinv)
        
        self.ell = X[:,0]
        self.n   = X[:,1]
        
        return X


if __name__ == '__main__':
    c_list  = np.array([ 0.5, 1.0, 1.5 , 2])   # wider spread, more of them
    x0_list = np.zeros_like(c_list)
    
    N = 2**8
    L = 20*2*np.pi

    T = 1
    M = 60*T # 30fps
    
    
    wp = Packet(L,N,T,M,c_list,x0_list)
    wp.make_solitons()
    wp.LSM()
    
    order = np.argsort(wp.k)
    k   = wp.k[order]
    ell = wp.ell[order]
    n   = wp.n[order]
    
    
    H_sorted = wp.H[:,0,order]
    # H_sorted_2 = wp.H[0,2,order]

    # print(wp.H.shape)

    # dhdt = wp.dHdt[0,3,order]
    # d_eta_x = -1j*wp.k[order]*wp.H[0,3,order]*wp.c_list[0]
    
    # plt.plot(wp.k[order], dhdt, label = 'dHdt')
    # plt.plot(wp.k[order], d_eta_x, alpha = 0.5 , linestyle = '--' , label = 'iku_x')
    # plt.legend()
    # plt.show()
    
    # for i, c in enumerate(c_list):
    #     plt.semilogy(wp.k[order], np.abs(wp.H[i, 0, :])[order], label=f'c={c}')
    # plt.xlabel(r'$\xi$'); plt.ylabel(r'$|\hat H(\xi)|$')
    # plt.legend(); plt.show()
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 4))

    idx = int(3*L/(4*np.pi)+1)   #    N//8
    a_neg = N//2 - idx
    a_pos = N//2 +idx +1
    
    ell_theory = 1j * k**3
    n_theory   = -3j* k

    axes[0,0].plot(k[a_neg:a_pos], ell[a_neg:a_pos].imag, '.', label='estimated Im($\\ell$)')
    axes[0,0].plot(k[a_neg:a_pos], ell_theory[a_neg:a_pos].imag, '-', label=r'theory: $\xi^3$')
    
    axes[0,0].set_xlabel(r'$\xi$')
    axes[0,0].set_ylabel(r'Im($\ell$)')
    axes[0,0].set_title(r'$\ell(\xi)$')
    axes[0,0].legend()

    axes[0,1].plot(k[a_neg:a_pos], n[a_neg:a_pos].imag, '.', label='estimated Im($n$)')
    axes[0,1].plot(k[a_neg:a_pos], n_theory[a_neg:a_pos].imag, '-', label=r'theory: $-3\xi$')
    axes[0,1].set_xlabel(r'$\xi$')
    axes[0,1].set_ylabel(r'Im($n$)')
    axes[0,1].set_title(r'$n(\xi)$')
    axes[0,1].legend()
    
    for i,c in enumerate(c_list):
        axes[1,0].plot(k[a_neg:a_pos], H_sorted[i, a_neg:a_pos].real, label=f'c={c}')
        axes[1,1].plot(k[a_neg:a_pos], H_sorted[i, a_neg:a_pos].imag, label=f'c={c}')
    axes[1,0].set_xlabel(r'$\xi$')
    axes[1,0].set_ylabel(r'Re($\eta$)')
    # axes[1,0].set_title(r'$H(\xi)$')
    axes[1,0].legend()
    
    axes[1,1].set_xlabel(r'$\xi$')
    axes[1,1].set_ylabel(r'Im($\eta$)')
    # axes[1,1].set_title(r'$n(\xi)$')
    axes[1,1].legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # animate_solution(k,wp.H[0,:,order].T)

    # print("max |Re(ell)|:", np.max(np.abs(ell.real)))
    # print("max |Re(n)|:  ", np.max(np.abs(n.real)))