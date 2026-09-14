# Integrating factor solution of Linear pdes
# 11.09.26 
# HSG

# Solving equations of the type u_t - Lu = 0
# Assuming the Fourier transformation of u exist for the initial condition u(0,*) = g(x)
# In the spectral space we then get u(t,x)= g(x)*e^{L*t} for any L 
import numpy as np
class linear_solver():
    """Solves linear PDEs with the Integrating Factor method"""
    def __init__(self, u0, L):
        self.u0         =  u0
        self.u0hat      = np.fft.fft(u0)
        self.N          = len(u0)
        self.k          = np.fft.fftfreq(self.N,d = L/self.N)*2*np.pi

    def solve(self,Lk, T:int , dt:float):
        self.num_steps  = max(int(T/dt) , 1)
        u_hat      = np.zeros((self.num_steps,self.N), dtype=complex)

        u_hat[0,:] = self.u0hat
        time_arr = np.linspace(dt, T, self.num_steps-1)
        u_hat[1:,:] = self.u0hat*np.exp(np.outer(time_arr,Lk(self.k)))


        return u_hat




    if __name__ == '__main__':
        pass
        