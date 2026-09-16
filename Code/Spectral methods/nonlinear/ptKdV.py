#Periodic traveling waves from the KdV equation
# Solves the problem (L-c)phi + 0.5 phi^2 = B
# c and phi are unknown but the amplitude and period and mean are given. 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# --- Problem setup ------------
N = 2**8
P = 2*np.pi
x = np.linspace(0,P,N, endpoint=False)
dx = x[1]-x[0]

# Fourier Frequencies
k = np.fft.fftfreq(N,d=dx)*2*np.pi

# KdV linear symbol
def m(k):
    return 1 - (k**2)/6

# Prescribed constraints
phi_mean_prescribed =   1.0
amp_prescribed =        0.5

# --- Residual ----------
def residual(phi, c, B):
    """Returns the residual of the PDE"""
    phi_hat =   np.fft.fft(phi)
    Lphi_hat =  m(k)*phi_hat
    Lphi =      np.fft.ifft(Lphi_hat).real
    return Lphi - c*phi + 0.5*phi**2 - B

# --- Full Nonlinear system ---
def F(unknown):
    """Stacking the residual and the constraints"""
    phi = unknown[:-2]
    c,B = unknown[-2], unknown[-1]
    res = residual(phi,c,B)
    mean_eq = phi.mean() - phi_mean_prescribed
    amp_eq = (phi.max() - phi.min())/2 - amp_prescribed

    return np.concatenate([res, [mean_eq, amp_eq]])
# --- Initial guess -------------
phi_init = phi_mean_prescribed + amp_prescribed*np.cos(x)
c_init = 1
B_init = 0

unknown_init = np.concatenate([phi_init,[c_init,B_init]])

# --- Solve -------------
sol = root(F, unknown_init)

phi_sol = sol.x[:-2]
c_sol, B_sol = sol.x[-2], sol.x[-1]

print("Converged:", sol.success)
print("wave speed c = ", c_sol)
print("Constant B = ", B_sol)

plt.plot(phi_sol)
plt.show()