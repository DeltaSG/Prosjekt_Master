# The KdV equation
# 14.09.26 
# HSG

# u_t + 6uu_x + u_xxx = 0

import numpy as np
from linear.help import *
from nonlinear.spectralNewton import solve_with_continuation



class KdV():
    """The KdV equation using 3/2 Padding for dealiasing the Nonlinearity"""
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
        # k_max = np.max(np.abs(k))
        # self.mask = (np.abs(k) < (2/3)*k_max)
        k_pad = np.zeros(3*self.N//2, dtype=complex)
        half = self.N//2
        k_pad[:half+1] = k[:half+1]
        k_pad[-(half-1):] = k[-(half-1):]
        self.k_pad = k_pad

        

    # def nonlinear_rhs(self, u_hat_in):
    #     """RHS = -F[u*u_x], pseudospectrally, with 2/3 dealiasing"""

    #     u = np.fft.ifft(u_hat_in*self.mask)
    #     ux = np.fft.ifft(1j * self.k*u_hat_in*self.mask)
    #     nl_hat = np.fft.fft((u * ux).real)
    #     # nl_hat = 0.5*1j*self.k*np.fft.fft((u*u).real)
    #     # nl_hat*=self.mask
    #     return -6*nl_hat

    def nonlinear_rhs(self, u_hat_in):
        Npad = 3*self.N//2
        u_hat_pad = np.zeros(Npad, dtype=complex)
        half = self.N//2
        u_hat_pad[:half+1] = u_hat_in[:half+1]
        u_hat_pad[-(half-1):] = u_hat_in[-(half-1):]     # careful with even/odd N, negative-freq block

        u_pad  = np.fft.ifft(u_hat_pad) * (Npad/self.N)
        ux_pad = np.fft.ifft(1j*self.k_pad*u_hat_pad) * (Npad/self.N)
        nl_pad = np.fft.fft((u_pad*ux_pad).real)

        if not np.all(np.isfinite(nl_pad)):
            print("nl_pad contains non-finite values")
            print("max u_pad =", np.max(np.abs(u_pad)))
            print("max ux_pad =", np.max(np.abs(ux_pad)))
            print("max u_hat =", np.max(np.abs(u_hat_in)))
            raise ValueError("nl_pad contains non finite values")

        nl_hat = np.zeros(self.N, dtype=complex)
        nl_hat[:half+1]  = nl_pad[:half+1] * (self.N/Npad)
        nl_hat[-(half-1):] = nl_pad[-(half-1):] * (self.N/Npad)

        rhs = -6*nl_hat

        # print(np.max(np.abs(u_hat_in)), np.max(np.abs(rhs)))
        return rhs

    def rk4_ETD(self, u_hat_in,E,E2, h):
        """ETD RK4 for IF"""
        # self.Lk = np.zeros_like(self.k)
        def G(w):
             return self.nonlinear_rhs(w)
        k1 = G(u_hat_in)
        k2 = G(E*u_hat_in + 0.5*h*k1)
        k3 = G(E*u_hat_in + 0.5*h*k2)
        k4 = G(E2*u_hat_in + E*h*k3)

        # print(
        #                 np.max(np.abs(k1)),
        #                 np.max(np.abs(k2)),
        #                 np.max(np.abs(k3)),
        #                 np.max(np.abs(k4))
        #             )
        
        return E2*u_hat_in + (h/6.0)*(E2*k1 + E*2*k2 + E*2*k3 + k4)

    def IF_rhs(self,v_hat,t):
         u_hat = np.exp(self.Lk*t)*v_hat
         return np.exp(-self.Lk*t)*self.nonlinear_rhs(u_hat)

    def IF_rk4_step(self, v_hat, t, h):
         k1 = self.IF_rhs(v_hat,t)
         k2 = self.IF_rhs(v_hat+0.5*h*k1, t+0.5*h)
         k3 = self.IF_rhs(v_hat+0.5*h*k2, t+0.5*h)
         k4 = self.IF_rhs(v_hat+h*k3, t+h)
         return v_hat + (h/6)*(k1+2*(k2+k3) + k4)

    def rk4_step(self, u_hat_in, h):
            """RK4"""
            k1 = self.nonlinear_rhs(u_hat_in)
            k2 = self.nonlinear_rhs(u_hat_in + 0.5*h*k1)
            k3 = self.nonlinear_rhs(u_hat_in + 0.5*h*k2) 
            k4 = self.nonlinear_rhs(u_hat_in + h*k3)

            
            return u_hat_in + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def integratingFactor(self):
        v_hat = self.u0_hat.copy()
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

            v_hat = self.IF_rk4_step(v_hat, t_current, h)
            t_current += h
            step_count += 1

            if step_count % self.save == 0 or t_current >= self.T - 1e-14:
                        u_hat = np.exp(self.Lk*t_current)*v_hat
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
    L = 4*np.pi
    sym = True
    x = np.linspace(-L/2, L/2, N, endpoint=False) if sym else np.linspace(0, L, N, endpoint=False)
    # u0 = np.exp(-(x**2)/2)
    c = 4
    x_0 = np.pi
    def u(x,c, x0):
        return (c/2) * 1/(np.cosh(np.sqrt(c)/2 *(x - x0))**2)

    u0_ = u(x,c,x_0)

    # u, mk, mask = solve_with_continuation(P=L, amp_target=1, N=N, mean = 0.0)

    # u0 = np.fft.irfft(u0_, N)
    be = KdV(u0_ , L=L , dt = 5e-4, T = 5, save_every=1)
    t,u_if = be.solve(method="SS", set=False) # Second order time acc
    # be.solve(method='SS', set=True) # Second order time acc

    be.conservation(method = "SS")

    animate_solution(x,u_if)
