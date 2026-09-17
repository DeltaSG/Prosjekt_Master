#Spectral Newton solver for finding traveling KdV waves 

import numpy as np 
import matplotlib.pyplot as plt 
def build_phi_hat(u,mean,amp,N):
    phi_hat =       np.zeros(N+1, dtype=complex)
    
    
    phi_hat =   u[:-2]
    # phi_hat[0] =    mean
    # phi_hat[1] =    amp/2
    c = u[-2]
    B = u[-1]
    return phi_hat, c, B

def physical_solution(phi_hat, N):
    return np.fft.irfft(phi_hat, n=N)

def nonlinear_hat(phi_hat,N):
    phi = physical_solution(phi_hat,N)
    nl = 0.5*phi**2
    return np.fft.rfft(nl)

def residual(u,mk,mean,amp,N):
    M = len(mk)-1
    phi_hat, c, B = build_phi_hat(
        u,mean, amp, M
    )
    
    nl_hat = nonlinear_hat(phi_hat,N)
    F = (mk-c)*phi_hat + nl_hat
    F[0] -= B*N
    
    mean_eq = phi_hat[0]/N - mean
    amp_eq = phi_hat[1]/N - amp/2
    return np.concatenate([
        np.real(F),
        [mean_eq],
        [amp_eq]
        ])


def jacobian_fd(u,mk,mean,amp,N, eps = 1e-8):
    """Naive FDM jacobian"""
    F0 = residual(u,mk,mean,amp,N)
    m = len(F0)
    
    J = np.zeros((m,m))
    for j in range(m):
        up = u.copy()
        up[j] += eps
        J[:,j] = (
            residual(up,mk,mean,amp,N)-F0
        )/eps
    return J

def newton(u0,mk,mean,amp,N,tol=1e-10, maxit = 50, verbose=True):
    u = u0.copy()
    
    for it in range(maxit):
        F = residual(u,mk,mean,amp,N)
        err = np.linalg.norm(F)
        if verbose:
            print(it,err)
        if err < tol:
            return u
        J = jacobian_fd(u,mk,mean,amp,N)
        delta = np.linalg.solve(J,-F)
        u += delta
        
    raise RuntimeError("No convergence")

if __name__ == '__main__':
    N = 2**8 # Full spectrum
    M = N//2 # Real spectrum
    amp = 1
    mean = 1
    P =     2*np.pi
    
    u0 = np.zeros(M+3)
    u0[0] = mean
    u0[1] = amp/2
    # Initial wave speed
    u0[-2] = 1- (2*np.pi/P)**2 /6
    #Initial B
    u0[-1] = 0.0
    
    x = np.linspace(0,P,N, endpoint=False)
    dx = P/N
    #rfft wave numbers
    k = 2*np.pi*np.fft.rfftfreq(N,dx)
    mk = 1- k**2/6
    
    u = newton(u0,mk,mean,amp,N)
    
    phi_hat = u[:-2]
    c_sol , B_sol = u[-2], u[-1]
    phi_sol = np.fft.irfft(phi_hat, N)
    print(f"amplitude = {(np.max(phi_sol) - np.min(phi_sol))/2}")

    
    print(f"Wave speed c= {c_sol}")
    print(f"Constant B = {B_sol}")
    
    plt.plot(x, phi_sol)
    plt.show()