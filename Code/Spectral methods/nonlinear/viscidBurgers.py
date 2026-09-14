# The Burgers' equation
# 14.09.26 
# HSG

import numpy as np
from ..linear.help import *



class Burgers():
    """The Burgers' equation using 2/3 masking for dealiasing the Nonlinearity"""
    def __init__(self,u0, epsilon:float , L:float, dt:float, safety_margin:float = 0.9, save_every:int = 1):
        self.u0         = u0
        self.u0_hat     = np.fft.fft(u0)
        min_slope  = np.min(
             np.fft.ifft(1j*self.u0_hat).real
                                    )
        self.N          = len(u0)
        self.epsilon    = epsilon
        self.L          = L
        dx              = L/N
        self.k          = np.fft.fftfreq(N,d=dx)*2*np.pi
        self.Lk         = -epsilon*self.k**2
        self.dt         = dt
        self.save       = save_every

        if min_slope >= 0:
             raise ValueError(
                "min(u0') >= 0: No compressive region, characteristics never cross."
                "i.e. No finite shock time exist for the given initial condition." 
             )
        self.ts = -1/min_slope
        self.t_target = self.ts*safety_margin

    def nonlinear_rhs(self, u_hat_in):
        """RHS = -F[u*u_x], pseudospectrally, with 2/3 dealiasing"""

        k = self.k
        k_max = np.max(np.abs(k))
        mask = (np.abs(k) < (2/3)*k_max)

        u = np.fft.ifft(u_hat_in).real
        ux = np.fft.ifft(1j * k*u_hat_in).real
        nl_hat = np.fft.fft(u * ux)
        nl_hat*=mask
        return -nl_hat

    def rk4_step(self, u_hat_in,E,E2, h):
        """ETD RK4 (?)"""
        k1 = self.nonlinear_rhs(u_hat_in)
        k2 = self.nonlinear_rhs(u_hat_in + 0.5*h*k1)
        k3 = self.nonlinear_rhs(u_hat_in + 0.5*h*k2) 
        k4 = self.nonlinear_rhs(E2*u_hat_in + E*h*k3)
        return E2*u_hat_in + (h/6.0)*(E2*k1 + E*2*k2 + E*2*k3 + k4)

    def integratingFactor(self):
        v_hat = self.u0_hat.copy()
        t_list = [0.0]
        u_list = [self.u0.copy()]

        t_current = 0.0
        step_count = 0

        E = np.exp(self.Lk*self.dt / 2)
        E2 = np.exp(self.Lk*self.dt)
        
        while t_current < self.t_target - 1e-14:
            h = min(self.dt , self.t_target- t_current) #Truncating final step
            if h != self.dt:
                E = np.exp(self.Lk*h/ 2)
                E2 = np.exp(self.Lk*h)

            v_hat = self.rk4_step(v_hat,h)
            t_current += h
            step_count += 1

            if step_count % self.save == 0 or t_current >= self.t_target - 1e-14:
                        t_list.append(self.t_target)
                        u_hat = np.exp(self.Lk * t_current) * v_hat
                        u_list.append(np.fft.ifft(u_hat).real)
                        
        return np.array(t_list), np.array(u_list), self.ts

    def solve(self, method:str = "IF"):
        """Solves the Burgers' equation with the given method"""
        if method == ' IF':
            t_array, u_array , ts = self.integratingFactor()

        return t_array, u_array, ts


if __name__ == '__main__':

    N = 2**8
    L = 2*np.pi
    sym = True
    x = np.linspace(-L/2, L/2, N, endpoint=False) if sym else np.linspace(0, L, N, endpoint=False)
    # u0 = np.exp(-(x**2)/2)
    u0 = np.sin(x)
    # u0 = u0_asymmetric_bump(x)

    be = Burgers(u0, 1 , L , dt = 1e-2 , safety_margin=0.9 , save_every=1)
    t,u,ts = be.solve()

    animate_solution(x,u)