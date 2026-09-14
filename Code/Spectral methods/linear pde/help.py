# Helper functions for pdes

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from lin_solver import linear_solver


def make_x_array(N:int , L:float , symmetric:bool) -> np.ndarray:
    x = np.linspace(
        -L/2, L/2, N, endpoint=False
        ) if symmetric else np.linspace(0,L,N,endpoint = False)
    return x


def animate_solution(x, u_hat):
    """Animates the solution u over the domain x"""
    N = u_hat.shape[1]
    # x = make_x_array(N, L, symmetric=symmetric)
    u = np.fft.ifft(u_hat, axis=1).real

    fig, ax = plt.subplots()
    t = 0 # Inital time

    ## Have not figured out how to represent it in the fourier space
    # line_fourier = ax.plot(
    #     x, u_hat[0,:],
    #     label = "Fourier"
    # )

    line_physical,  = ax.plot(
        x, u[0,:],
        label = "Physical",
        lw = 2
    )

    steps = u.shape[0]
    stride = max(steps//200 , 1)
    frame_indices = np.arange(0,steps-1, stride)


    def update(i):
        n = frame_indices[i]
        line_physical.set_ydata(u[n])
        return line_physical,

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        repeat = True,
        interval = 40,
        blit = False,
        )

    plt.show()



if __name__ == '__main__':
    L = 50
    N = 2**9
    T = 20
    dt = 1e-2
    x = make_x_array(N,L,True)
    u0 = np.exp(-x**2 / 5)

    init = linear_solver(u0,L)
    Lk = lambda k: 1j*k**3
    u_hat = init.solve(Lk,T,dt)

    animate_solution(x,u_hat)




