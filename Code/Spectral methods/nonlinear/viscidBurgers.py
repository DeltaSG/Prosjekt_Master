# The Burgers' equation
# 14.09.26 
# HSG

import numpy as np
from linear.help import *
from linear.lin_solver import LinearSolver



class Burgers():
    """The Burgers' equation using 2/3 masking for dealiasing the Nonlinearity"""
    def __init__(self,u0, epsilon:float , L:float, dt:float, T:int = 10, safety_margin:float = 0.9, save_every:int = 1):
        self.u0         = u0
        self.u0_hat     = np.fft.fft(u0)
        
        self.N          = len(u0)
        self.epsilon    = epsilon
        self.L          = L
        dx              = L/N
        self.k          = np.fft.fftfreq(N,d=dx)*2*np.pi
        self.Lk         = -epsilon*self.k**2
        self.dt         = dt
        self.save       = save_every

        min_slope  = np.min(
                     np.fft.ifft(
                          1j*self.k*self.u0_hat
                          ).real
                                            )

        if min_slope >= 0:
             raise ValueError(
                "min(u0') >= 0: No compressive region, characteristics never cross."
                "i.e. No finite shock time exist for the given initial condition." 
             )
        self.ts = -1/min_slope
        self.t_target = min(self.ts*safety_margin,T)

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

    def rk4_ETD(self, u_hat_in,E,E2, h):
        """ETD RK4 for IF"""
        k1 = self.nonlinear_rhs(u_hat_in)
        k2 = self.nonlinear_rhs(E*(u_hat_in + 0.5*h*k1))
        k3 = self.nonlinear_rhs(E*(u_hat_in + 0.5*h*k2)) 
        k4 = self.nonlinear_rhs(E2*u_hat_in + E*h*k3)
        return E2*u_hat_in + (h/6.0)*(E2*k1 + E*2*k2 + E*2*k3 + k4)

    def rk4_step(self, u_hat_in, h):
            """RK4"""
            k1 = self.nonlinear_rhs(u_hat_in)
            k2 = self.nonlinear_rhs(u_hat_in + 0.5*h*k1)
            k3 = self.nonlinear_rhs(u_hat_in + 0.5*h*k2) 
            k4 = self.nonlinear_rhs(u_hat_in + h*k3)
            return u_hat_in + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def integratingFactor(self):
        u_hat = self.u0_hat.copy()
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

            u_hat = self.rk4_ETD(u_hat, E, E2,h)
            t_current += h
            step_count += 1

            if step_count % self.save == 0 or t_current >= self.t_target - 1e-14:
                        t_list.append(t_current)
                        u_list.append(np.fft.ifft(u_hat).real)
                        
        return np.array(t_list), np.array(u_list), self.ts

    def splitStep(self):
        u_hat = self.u0_hat.copy()
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

            # u_ = u_hat*E
            # u__ = self.rk4_step(u_, h)
            # u_hat = u_hat*E

            u_hat = E*self.rk4_step(u_hat*E,h)

            t_current += h
            step_count += 1

            if step_count % self.save == 0 or t_current >= self.t_target - 1e-14:
                        t_list.append(t_current)
                        u_list.append(np.fft.ifft(u_hat).real)
                        
        return np.array(t_list), np.array(u_list), self.ts
         

    def solve(self, method:str = "IF"):
        """Solves the Burgers' equation with the given method"""
        if method == 'IF':
            t_array, u_array , ts = self.integratingFactor()

        else: 
             t_array, u_array , ts = self.splitStep()

        return t_array, u_array, ts


if __name__ == '__main__':

    N = 2**8
    L = 4*np.pi
    sym = True
    x = np.linspace(-L/2, L/2, N, endpoint=False) if sym else np.linspace(0, L, N, endpoint=False)
    # u0 = np.exp(-(x**2)/2)
    u0 = np.sin(x)

    be = Burgers(u0, epsilon=1e-1 , L=L , dt = 1e-2, T = 2 , safety_margin=1.1 , save_every=1)
    t,u_if,ts = be.solve(method="IF") # Fourth order time acc
    t,u_ss,ts = be.solve(method='SS') # Second order time acc

    animate_solution(x,u_if)

    # for dt in [1e-1, 5e-2, 2.5e-2, 1.25e-2]:
    #     be = Burgers(u0, epsilon=1, L=L, dt=dt, T=1, safety_margin=0.9, save_every=1)
    #     t, u_if, ts = be.solve(method="IF")
    #     t2, u_ss, ts2 = be.solve(method="SS")
    #     diff = np.max(np.abs(u_if - u_ss))
    #     print(f"dt={dt:.4g}  max diff={diff:.4e}")