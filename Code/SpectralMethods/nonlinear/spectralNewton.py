#Spectral Newton solver for finding traveling KdV waves 

import numpy as np 
import matplotlib.pyplot as plt 
def build_phi_hat(u,N):
    phi_hat =       np.zeros(N+1, dtype=complex)
    
    phi_hat =   u[:-2]
    # phi_hat[0] =    mean
    # phi_hat[1] =    amp/2
    c = u[-2]
    B = u[-1]
    return phi_hat, c, B

def physical_solution(phi_hat, N):
    return np.fft.irfft(phi_hat, n=N)

def nonlinear_hat(phi_hat,N, mask):
    phi = physical_solution(phi_hat*mask,N)
    nl = 0.5*phi**2
    return np.fft.rfft(nl)

def residual(u,mk,mask,mean,amp,N):
    M = len(mk)-1
    phi_hat, c, B = build_phi_hat(
        u, M
    )
    
    nl_hat = nonlinear_hat(phi_hat,N, mask)
    F = (mk-c)*phi_hat + nl_hat
    F[0] -= B*N
    
    mean_eq =   phi_hat[0] - N*mean
    amp_eq =    phi_hat[1] - N*amp/2
    return np.concatenate([
        np.real(F),
        [mean_eq],
        [amp_eq]
        ])


def jacobian_fd(u,mk,mask,mean,amp,N, eps = 1e-8):
    """Naive FDM jacobian"""
    F0 = residual(u,mk,mean,amp,N)
    m = len(F0)
    
    J = np.zeros((m,m))
    for j in range(m):
        up = u.copy()
        up[j] += eps
        J[:,j] = (
            residual(up,mk,mask,mean,amp,N)-F0
        )/eps
    return J

def jv(v_hat, phi_hat, c ,mk, mask,N):
    phi = np.fft.irfft(phi_hat, n=N)
    v = np.fft.irfft(v_hat, n=N)
    return(
        (mk-c)*v_hat + np.fft.rfft(phi*v)*mask
    )
    
def jacobian(u, mk,mask, N):
    phi_hat = u[:-2]
    c = u[-2]
    M = len(phi_hat) - 1
    J = np.zeros((M+3,M+3))
    
    for j in range(M+1):
        e = np.zeros(M+1)
        e[j] = 1.0
        J[:M+1,j] = np.real(
            jv(e,phi_hat,c,mk,mask,N)
        )
    # dF/dc
    J[:M+1,-2] = -np.real(phi_hat)
    # dF/dB
    J[0,-1]= -N
    # amplitude and mean constraints
    J[M+1,0] = 1
    J[M+2,1] = 1
    return J

def linesearch(u,delta,err,mk,mask, mean, amp, N):
    alpha = 1
    while alpha > 1e-8:
        u_trial = u+ alpha*delta
        if np.linalg.norm(
            residual(u_trial,mk,mask,mean,amp,N)
        ) < err: break
        
        alpha*= 0.5
    return u_trial

def newton(u0,mk,mask,mean,amp,N,tol=1e-10, maxit = 50, verbose=True):
    u = u0.copy()
    
    for it in range(maxit):
        F = residual(u,mk,mask,mean,amp,N)
        err = np.linalg.norm(F)
        if verbose:
            print(it,err)
        if err < tol:
            return u
        # J = jacobian_fd(u,mk,mean,amp,N)
        J = jacobian(u,mk,mask,N)
        # if verbose:
        #     print("Testing analytical jacobian")
        #     J_fd = jacobian_fd(u,mk,mean,amp,N)
        #     print(
        #         np.linalg.norm(J_fd- J) / np.linalg.norm(J_fd)
        #     )
        delta = np.linalg.solve(J,-F)
        u = linesearch(u, delta, err, mk,mask, mean, amp, N)
        
    raise RuntimeError("No convergence")

def solve_with_continuation(P, amp_target, N, mean=0.0, n_steps=50,mask=None, **newton_kwargs):
    M = N//2
    dx = P/N
    k = 2*np.pi*np.fft.rfftfreq(N, dx)
    mk = 1 - k**2/6
    if mask is None:
        mask = (np.abs(k) < (2/3)*np.max(np.abs(k)))
    

    u = np.zeros(M+3)
    u[-2] = 1 - (2*np.pi/P)**2/6   # linear guess, valid at amp -> 0

    for amp in np.linspace(amp_target/n_steps, amp_target, n_steps):
        u[1] = N*amp/2             # move the target for this step
        u = newton(u,mk,mask,mean,amp,N,verbose=False)

    return u, mk, mask

if __name__ == '__main__':
    N =     2**8 # Full spectrum
    M =     N//2 # Real spectrum
    amp =   1
    mean =  0.0
    P =     8*np.pi
    
    u0 = np.zeros(M+3)
    u0[0] = N*mean
    u0[1] = N*amp/2
    # Initial wave speed
    u0[-2] = 1- (2*np.pi/P)**2 /6
    #Initial B
    u0[-1] = 0.0
    
    x = np.linspace(0,P,N, endpoint=False)
    dx = P/N
    #rfft wave numbers
    k = 2*np.pi*np.fft.rfftfreq(N,dx)
    mk = 1- k**2/6
    mask = (np.abs(k) < (2/3)*np.max(np.abs(k)))

    
    u, mk,mask = solve_with_continuation(P=P, amp_target=amp, N=N, mean = mean)
    # u = newton(u0,mk,mask,mean,amp,N)
    phi_hat = u[:-2]
    c_sol , B_sol = u[-2], u[-1]
    phi_sol = np.fft.irfft(phi_hat, N)
    print(f"amplitude = {(np.max(phi_sol) - np.min(phi_sol))/2}")

    
    print(f"Wave speed c= {c_sol}")
    print(f"Constant B = {B_sol}")
    
    plt.plot(x, phi_sol)
    plt.show()