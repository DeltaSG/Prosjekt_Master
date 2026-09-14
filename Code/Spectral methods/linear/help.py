# Helper functions for pdes

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Slider


def make_x_array(N:int , L:float , symmetric:bool) -> np.ndarray:
    x = np.linspace(
        -L/2, L/2, N, endpoint=False
        ) if symmetric else np.linspace(0,L,N,endpoint = False)
    return x


def animate_solution(x, u, interval = 40):
    """Animates the solution u over the domain x"""
    N = u.shape[1]
    # x = make_x_array(N, L, symmetric=symmetric)
    # u = np.fft.ifft(u_hat, axis=1).real

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ax_speed = plt.axes([0.2, 0.1, 0.6, 0.03])
    ax_stride = plt.axes([0.2, 0.15, 0.6, 0.03])

    speed_slider = Slider(
        ax = ax_speed,
        label='Speed',
        valmin=0.1,
        valmax=2,
        valinit=1
        )

    step_slider = Slider(
        ax = ax_stride,
        label='Stride',
        valmin=1,
        valmax=50,
        valinit=10,
        valstep=1
    )
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



    current_frame = [0]
    def update(i):
        stride = int(step_slider.val)
        speed = speed_slider.val
        current_frame[0]+= (speed *stride)
        n = int(current_frame[0]) % len(u)
        line_physical.set_ydata(u[n])
        return line_physical,

    anim = FuncAnimation(
        fig,
        update,
        repeat = True,
        interval = interval,
        blit = False,
        )

    
    plt.show()



if __name__ == '__main__':
    pass




