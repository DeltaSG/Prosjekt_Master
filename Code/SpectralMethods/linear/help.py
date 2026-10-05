# Helper functions for pdes

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Slider
from IPython.display import HTML
import pandas as pd

def make_x_array(N:int , L:float , symmetric:bool) -> np.ndarray:
    x = np.linspace(
        -L/2, L/2, N, endpoint=False
        ) if symmetric else np.linspace(0,L,N,endpoint = False)
    return x

def conservationOnL(N,L,f= lambda x: 0.5*np.cos(2*x),g = lambda x: np.sin(x),Lk = lambda k: 1 - (k**2)/6):
    """Testing inner product property of L.
    N: Number of points in the array.
    L: Domain of the array (symmetric).
    f,g: Test-functions.
    Lk: The spectral linear operator as a function of k.  """

    x = make_x_array(N,L,symmetric=True)
    dx = x[1]-x[0]

    k = np.fft.fftfreq(N,d = dx)*2*np.pi

    mk = Lk(k)

    u = np.fft.fft(f(x))
    v = np.fft.fft(g(x))

    s1 = np.inner(mk*u, v)
    s2 = np.inner(u, mk*v)
    assert np.allclose(s1,s2, atol=1e-12) , "The Linear operator (Lk) is not self adjoint"
    assert np.all(np.isreal(mk)), "The Linear operator (Lk) is not real"
    return 0

def pad(u_hat, N):
    """Zero-pad a spectrum (..., N) to (..., 3N/2) along the last axis."""
    Npad = 3*N//2
    half = N//2
    out = np.zeros(u_hat.shape[:-1] + (Npad,), dtype=complex)
    out[..., :half]        = u_hat[..., :half]
    out[..., -(half-1):]   = u_hat[..., -(half-1):]   # Nyquist mode dropped
    return out

def animate_solution(x, u, interval = 40):
    """Animates the solution u over the domain x"""
    N = u.shape[1]
    # x = make_x_array(N, L, symmetric=symmetric)
    # u = np.fft.ifft(u_hat, axis=1).real

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    ax.grid(True)

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



    current_frame = [0.0]
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
        frames= len(u),
        repeat = True,
        interval = interval,
        blit = False,
        )
  
    # plt.close(fig)
    return anim




def animate_solution_nb(x, u, stride=10, interval=40):
    """Plays the solution u (n_frames, N) over x once, in a notebook."""
    frames = range(0, u.shape[0], stride)
    pad = 0.05 * (u.max() - u.min() + 1e-12)

    fig, ax = plt.subplots()
    ax.set_ylim(u.min() - pad, u.max() + pad)
    ax.grid(True)
    line, = ax.plot(x, u[0], lw=2)

    def update(n):
        line.set_ydata(u[n])
        ax.set_title(f"frame {n}")
        return line,

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, repeat=False)
    plt.close(fig)  # stops the static figure from showing up as well
    return HTML(anim.to_jshtml(default_mode="once"))

def mass(u,dx):
    U = np.asarray(u)
    return dx*np.sum(U, axis=-1)

def l2(u,dx):
    U = np.asarray(u)
    return dx*np.sum(U**2, axis=-1)

def hamiltonian(u,mk,dx):
    """Working using Lu_x -> for KdV L = 1-k**2/6"""
    U = np.asarray(u)
    u_hat = np.fft.fft(U,axis=-1)
    lu_hat = mk*u_hat
    lu = np.fft.ifft(lu_hat, axis=-1)

    H2 = 0.5*dx*np.sum(U*lu, axis=-1)
    H3 = (dx/6)*np.sum(U**3, axis = -1)

    return H2 + H3


def fit_table(ell, n, part="imag"):
    """DataFrame of polyfit coefficients, columns = powers (highest first).
    ell, n: coefficient arrays from np.polyfit. part: 'imag', 'real' or 'both'."""
    deg = len(ell) - 1
    cols = [f"ξ^{p}" for p in range(deg, -1, -1)]

    if part == "both":
        return pd.DataFrame({"ell": ell, "n": n}, index=cols).T
    get = np.imag if part == "imag" else np.real
    return pd.DataFrame([get(ell), get(n)], index=["ell", "n"], columns=cols)

def fineprint(ell, n, part="imag"):
    deg = len(ell) - 1
    print(f"Fitting polynomial of degree {deg} to the symbol data...")
    print(fit_table(ell, n, part).to_string(float_format=lambda v: f"{v:.3f}"))

def conservationInfo(unit, name:str = "Mass"):
    print(5*"-","Conservation of ", name , 5*"-" )
    print(f"Initial {name}: {unit[0].real:.3f}")
    print(f"Final {name}: {unit[-1].real:.3f}")
    print(f"Maximum error: {(np.max(np.abs(unit -unit[0]).real)):.3f}")
    print(f"Relative error: {((np.max(np.abs(unit -unit[0]))/np.abs(unit[0]).real)):.3f}")
    print(30*"-")

    

if __name__ == '__main__':
    N = 2**5
    L = 2*np.pi 
    conservationOnL(N,L)




