# The KdV equation
# 14.09.26 
# HSG

# u_t + 6uu_x + u_xxx = 0

import numpy as np
from linear.help import *



class KdV():
    """The KdV equation using 2/3 masking for dealiasing the Nonlinearity"""
    def __init__(self,u0 , L:float, dt:float, T:int = 10, save_every:int = 1):
        self.u0         = u0
        self.u0_hat     = np.fft.fft(u0)
        
        self.N          = len(u0)
        self.L          = L
        dx              = L/self.N
        self.k          = np.fft.fftfreq(self.N,d=dx)*2*np.pi
        self.Lk         = 1j*(self.k**3)
        self.dt         = dt
        self.T          = T
        self.save       = save_every

        k = self.k
        k_max = np.max(np.abs(k))
        self.mask = (np.abs(k) < (2/3)*k_max)
        

    def nonlinear_rhs(self, u_hat_in):
        """RHS = -F[u*u_x], pseudospectrally, with 2/3 dealiasing"""

        u = np.fft.ifft(u_hat_in)
        ux = np.fft.ifft(1j * self.k*u_hat_in)
        nl_hat = np.fft.fft((u * ux).real)
        nl_hat*=self.mask
        return -6*nl_hat

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
        
        while t_current < self.T - 1e-14:
            h = min(self.dt , self.T- t_current) #Truncating final step
            if h != self.dt:
                E = np.exp(self.Lk*h/ 2)
                E2 = np.exp(self.Lk*h)

            u_hat = self.rk4_ETD(u_hat, E, E2,h)
            t_current += h
            step_count += 1

            if step_count % self.save == 0 or t_current >= self.T - 1e-14:
                        t_list.append(t_current)
                        u_list.append(np.fft.ifft(u_hat).real)
                        
        return np.array(t_list), np.array(u_list)

    def splitStep(self):
        u_hat = self.u0_hat.copy()
        t_list = [0.0]
        u_list = [self.u0.copy()]

        t_current = 0.0
        step_count = 0

        E = np.exp(self.Lk*self.dt / 2)
        E2 = np.exp(self.Lk*self.dt)
        
        while t_current < self.T - 1e-14:
            h = min(self.dt , self.T- t_current) #Truncating final step
            if h != self.dt:
                E = np.exp(self.Lk*h/ 2)
                E2 = np.exp(self.Lk*h)

            # u_ = u_hat*E
            # u__ = self.rk4_step(u_, h)
            # u_hat = u_hat*E

            u_hat = E*self.rk4_step(u_hat*E,h)

            t_current += h
            step_count += 1

            if step_count % self.save == 0 or t_current >= self.T - 1e-14:
                        t_list.append(t_current)
                        u_list.append(np.fft.ifft(u_hat).real)
                        
        return np.array(t_list), np.array(u_list)

    def solve(self, method:str = "IF", set = True):
        """Solves the KdV' equation with the given method"""
        # The code for the solvers can be merged and then it is just in the while loop one needs to figure out what method to use. 
        # eg f(x) = IF if method == if elif method == ss f(x) = SS. This might not have any run time improvement but is easier to look at
        if method == 'IF':
            t_array, u_array = self.integratingFactor()

        else: 
             t_array, u_array = self.splitStep()

        if set:
             self.t_array = t_array
             self.u =       u_array
        else:
            return t_array, u_array

    def conservation(self, method = "IF"):
        if not hasattr(self, "u"):
            print("Solving PDE for u using: ", method)
            self.solve(method=method, set= True)

        dx = self.L/self.N
        u = self.u
        #Mass
        m = mass(u,dx)
        conservationInfo(m,"Mass")
        # L2
        l = l2(u,dx)
        conservationInfo(l, "L2")
        #Hamiltonian
        l = lambda k: 1-k**2/6
        mk = l(self.k)
        h = hamiltonian(u,mk,dx)
        conservationInfo(h, "Hamiltonian")



if __name__ == '__main__':

    N = 2**8
    L = 40
    sym = True
    x = np.linspace(-L/2, L/2, N, endpoint=False) if sym else np.linspace(0, L, N, endpoint=False)
    # u0 = np.exp(-(x**2)/2)
    c = 4
    x_0 = np.pi
    def u(x):
        return (c/2) * 1/(np.cosh(np.sqrt(c)/2 *(x - x_0))**2)

    u0 = u(x)

    be = KdV(u0 , L=L , dt = 1e-3, T = 5, save_every=1)
    t,u_if = be.solve(method="IF", set=False) # Fourth order time acc
    be.solve(method='SS', set=True) # Second order time acc

    # be.conservation(method = "SS")

    animate_solution(x,u_if)
