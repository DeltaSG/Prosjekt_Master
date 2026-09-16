# Newton solver with jacobian
# 15.09.26
# HSG

import numpy as np 

def build_jacobian_(phi, c, L_mat):
    
    J = np.zeros((N+2,N+2))
    J[:N, :N] =     L_mat - c*np.eye(N) + np.diag(phi)
    J[:N,N] =       -phi # d(res)/dc
    J[:N,N+1] =     -1 # d(res)/dB
    J[N,:N] =       1/N # d(mean_eq)/dphi
    imax, imin =    np.argmax(phi), np.argmin(phi)
    J[N+1,imax] +=  0.5
    J[N+1, imin] -= 0.5
    return J

def F_(unknown):
    phi = unknown[:-2]
    c,B = unknown[-2], unknown[-1]
    res = residual(phi,c,B)
    mean_eq = phi.mean() - phi_mean_prescribed
    amp_eq  = (phi.max() - phi.min())/2 - amp_prescribed
    return np.concatenate([res, [mean_eq, amp_eq]])

def newton_solver(u0, tol=1e-10, maxit=50):
    u = u0.copy()
    for it in range(maxit):
        phi, c, B = u[:-2], u[-2], u[-1]
        Fu = F_(u)
        
        if np.linalg.norm(Fu) < tol:
            return u, it
        J = build_jacobian_(phi,c)
        delta = np.linalg.solve(J, -Fu)
        u += delta
    raise RuntimeError("No Convergence")

def m_kdv(k):
    return 1-(k**2)/6

def build_grid_andL(N,P,symbol):
    x = np.linspace(0,P,N,endpoint=False)
    dx = x[1]-x[0]
    k = np.fft.fftfreq(N,d=dx)*2*np.pi
    mk = symbol(k)
    
    L_mat = np.real(
            np.fft.ifft(
                mk[:,None] * np.fft.fft(
                    np.eye(N), axis = 0
                ), axis = 0
            )
        )
    
    return x, k, mk, L_mat

def residual(phi, c, B, mk):
    phi_hat = np.fft.fft(phi)
    Lphi = np.fft.ifft(mk*phi_hat).real
    return Lphi - c*phi + 0.5*phi**2 - B