# Sanity check for the spectral solvers.
# HSG 06.10

from nonlinear.spectralSolvers import *

N = 2**8
amp = 0.3
mean = 0.0
P = 16*np.pi

kdv = KdVSolver(P=P, target_amp= amp, mean= mean , N = N)
kdv.solve_w_continuation(n_steps=50)
print("KdV: c = ", kdv.c, "B = ", kdv.B)

# whitham = WhithamSolver(P, amp, mean, N)
# whitham.solve_w_continuation()

# fkdv = fKdVSolver(P,amp, mean, N, alpha= 2) # Same as KdV
# fkdv.solve_w_continuation()


def check(solver, name=""):
    ph = solver.phi_hat                       # rfft coefficients, length M+1
    print(name)
    print("  zero mode      :", ph[0])        # should be ~1e-15
    print("  max |imag|     :", np.max(np.abs(np.imag(ph))))
    print("  min of k>=1    :", ph[1:].min()) # should be > 0
    print("  all positive   :", np.all(ph[1:] > 0))
    print("  phi_hat[1]/N*2 :", ph[1]*2/solver.N, "(target amp:", solver.target_amp, ")")

check(kdv, "KdV");# check(whitham, "Whitham"); check(fkdv, "fKdV")

from scipy.optimize import brentq

def soliton_hat_cp(solver, cp):
    A, kappa = 3*cp, np.sqrt(3*cp/2)
    k = solver.k
    F = np.zeros_like(k)
    F[1:] = A*np.pi*k[1:]/kappa**2 / np.sinh(np.pi*k[1:]/(2*kappa))
    return solver.N/solver.P * F

cp = brentq(lambda cp: soliton_hat_cp(kdv, cp)[1] - kdv.phi_hat[1], 1e-6, 10)
print("fitted cp:", cp, " solver c:", kdv.c, " 1+cp-6cp/(kappa*P):", 
      1 + cp - 6*cp/(np.sqrt(3*cp/2)*kdv.P))


def soliton_hat(solver):
    cp    = solver.c - 1                      # KdV only: speed above 1
    A     = 3*cp
    kappa = np.sqrt(3*cp/2)
    k     = solver.k
    F     = np.empty_like(k)
    F[1:] = A*np.pi*k[1:]/kappa**2 / np.sinh(np.pi*k[1:]/(2*kappa))
    F[0]  = 0.0                               # mean removed
    return solver.N/solver.P * F

# plt.semilogy(kdv.k[1:], kdv.phi_hat[1:], 'o-', label="KdV solver")
# plt.semilogy(kdv.k[1:], soliton_hat(kdv)[1:], 'x--', label="soliton (analytic)")
# plt.legend(); plt.show()

def compare_to_soliton(solver, cp, name="KdV"):
    N, P, k = solver.N, solver.P, solver.k
    x = np.arange(N) * P / N                         # same grid as the solver

    ph_solver = solver.phi_hat / N                   # normalise so N drops out
    ph_sol    = soliton_hat_cp(solver, cp) / N

    phi_solver = nf.irfft(solver.phi_hat, n=N)
    phi_sol    = nf.irfft(soliton_hat_cp(solver, cp), n=N)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Fourier space, semilog, skipping the zero mode
    ax[0].semilogy(k[1:], np.abs(ph_solver[1:]), 'o-',  label=f"{name} solver")
    ax[0].semilogy(k[1:], np.abs(ph_sol[1:]),    'x--', label="soliton (analytic)")
    ax[0].set_xlabel("k"); ax[0].set_title(r"$|\hat\phi_k|/N$")

    # 2. Physical space, shifted so the peak sits in the middle
    xs = (x + P/2) % P - P/2
    idx = np.argsort(xs)
    ax[1].plot(xs[idx], phi_solver[idx], label=f"{name} solver")
    ax[1].plot(xs[idx], phi_sol[idx], '--', label="soliton (periodised)")
    ax[1].set_xlabel("x"); ax[1].set_title(r"$\phi(x)$")

    # 3. Error per mode
    ax[2].semilogy(k[1:], np.abs(ph_solver[1:] - ph_sol[1:]) + 1e-18, 'o-')
    ax[2].set_xlabel("k"); ax[2].set_title("mode-wise difference")

    for a in ax: a.grid(True)
    ax[0].legend(); ax[1].legend()
    plt.tight_layout(); plt.show()

    print("zero mode          :", solver.phi_hat[0])
    print("min of modes k>=1  :", solver.phi_hat[1:].min())
    print("solver c           :", solver.c)

compare_to_soliton(kdv, cp)