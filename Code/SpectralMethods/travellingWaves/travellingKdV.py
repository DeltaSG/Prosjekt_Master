import numpy as np
import matplotlib.pyplot as plt

from nonlinear.spectralSolvers import KdVSolver
from nonlinear.solvers import KdV
from linear.help import *

# Creating a travelling wave:
P = 10*2*np.pi
N = 2**8

amp_target = 0.10
mean = 0.0

kdv = KdVSolver(P=P, target_amp=amp_target, N=N, mean=mean)
kdv.solve_w_continuation()
u = kdv.u
phi_hat = u[:-2]
c_sol, B_sol = u[-2], u[-1]

# kdv._plot_()

amp_target = 0.25
mean = 0.0

whit = KdVSolver(P=P, target_amp=amp_target, N=N, mean=mean)
whit.solve_w_continuation()
u = whit.u
phi2_hat = u[:-2]
c_sol, B_sol = u[-2], u[-1]

# Solving it forward in time
kdvs = KdV(P=P, dt = 5e-3, T=30, u0=np.fft.irfft(phi_hat + phi2_hat, n=N), save_every=2)
t,u = kdvs.splitStep()
x = np.linspace(0,P,N, endpoint=False)
anim = animate_solution(x, u)
plt.show()