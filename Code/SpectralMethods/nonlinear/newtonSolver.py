#Periodic traveling waves from the KdV equation
# Solves the problem (L-c)phi + 0.5 phi^2 = B
# c and phi are unknown but the amplitude and period and mean are given. 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# --- Problem setup ------------
N = 2**8
P = 8*np.pi


# KdV linear symbol
def m_kdv(k):
    return 1 - (k**2)/6

# Prescribed constraints
phi_mean_prescribed =   0.0
amp_prescribed =        0.1

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

x,k,mk,L_mat = build_grid_andL(N,P,m_kdv)

# --- Residual ----------
def residual(phi, c, B, mk):
    """Returns the residual of the PDE"""
    phi_hat =   np.fft.fft(phi)
    Lphi_hat =  mk*phi_hat
    Lphi =      np.fft.ifft(Lphi_hat).real
    return Lphi - c*phi + 0.5*phi**2 - B

# --- Initial guess -------------
phi_init = phi_mean_prescribed + amp_prescribed*np.cos(2*np.pi*x/P)
c_init = 1 #- (2*np.pi/P)**2 /6
B_init = 0

unknown_init = np.concatenate([phi_init,[c_init,B_init]])

# --- Solve -------------
def build_jacobian_(phi, c, L_mat):
    
    J = np.zeros((N+2,N+2))
    J[:N, :N] =     L_mat - c*np.eye(N) + np.diag(phi) # d(res)/dphi
    J[:N,N] =       -phi # d(res)/dc
    J[:N,N+1] =     -1 # d(res)/dB
    J[N,:N] =       1/N # d(mean_eq)/dphi
    # imax, imin =    np.argmax(phi), np.argmin(phi)
    # J[N+1,imax] +=  0.5
    # J[N+1, imin] -= 0.5
    j = np.arange(N)
    J[N+1,:N] = np.cos(2*np.pi*j/N)/N
    J[N+1,N] = 0
    J[N+1,N+1]=0
    
    return J

def F_(unknown):
    phi = unknown[:-2]
    c,B = unknown[-2], unknown[-1]
    res = residual(phi,c,B, mk=mk)
    mean_eq = phi.mean() - phi_mean_prescribed
    # amp_eq = (np.max(phi)- np.min(phi))/2 - amp_prescribed
    phi_hat = np.fft.fft(phi)/N
    amp_eq  = np.real(phi_hat[1])- amp_prescribed/2
    return np.concatenate([res, [mean_eq, amp_eq]])

def newton_solver(u0, tol=1e-10, maxit=50):
    u = u0.copy()
    for it in range(maxit):
        phi, c, B = u[:-2], u[-2], u[-1]
        Fu = F_(u)
        
        if np.linalg.norm(Fu) < tol:
            return u, it
        J = build_jacobian_(phi,c,L_mat)
        print(it,r"$||Fu||_2$:" ,np.linalg.norm(Fu))
        print(r"$Cond(J)$:",np.linalg.cond(J))
        delta = np.linalg.solve(J, -Fu)
        u += delta
    raise RuntimeError("No Convergence")

sol , it = newton_solver(unknown_init)

phi_sol = sol[:-2]
c_sol, B_sol = sol[-2], sol[-1]

print("Converged after #", it , "iterations")
print("wave speed c = ", c_sol)
print("Constant B = ", B_sol)

plt.plot(x,phi_sol)

plt.show()

phi_hat = np.fft.fft(phi_sol)/N
for n in range(15):
    print(n, abs(phi_hat[n]))