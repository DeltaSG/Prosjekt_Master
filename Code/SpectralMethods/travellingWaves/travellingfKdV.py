import numpy as np
import matplotlib.pyplot as plt

from nonlinear.spectralSolvers import fKdVSolver
from Code.SpectralMethods.nonlinear.solvers import fKdV
from linear.help import *

# Creating a travelling wave:
P = 10 *2*np.pi
N = 2**8
alpha = -1/2

amp_target = 0.05
mean = 0.0

fkdv = fKdVSolver(P=P, target_amp=amp_target, N=N, mean=mean, alpha=alpha)
fkdv.solve_w_continuation()
u = fkdv.u
phi_hat = u[:-2]
c_sol, B_sol = u[-2], u[-1]

amp_target2 = 0.15
mean2 = 0.05

fkdv = fKdVSolver(P=P, target_amp=amp_target2, N=N, mean=mean2, alpha=alpha)
fkdv.solve_w_continuation()
u = fkdv.u
phi2_hat = u[:-2]
c2_sol, B2_sol = u[-2], u[-1]

# kdv._plot_()

# Solving it forward in time
fkdvs = fKdV(P=P, dt = 5e-3, T=40, u0=np.fft.irfft(phi_hat+ phi2_hat, n=N), save_every=2, alpha=alpha)
t,u = fkdvs.splitStep()
x = np.linspace(0,P,N, endpoint=False)
anim = animate_solution(x, u)
plt.show()