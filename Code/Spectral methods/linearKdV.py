# Solver for the linear KdV equation using spectral methods
def spectral_linear_kdv_solver(u0_hat, L, T, dt, method= 'analytic'):
    """
    Solve the linear KdV equation using spectral methods.

    Parameters:
    u0_hat : array_like
        The initial condition in Fourier space.
    L : float
        The length of the spatial domain.
    T : float
        The total time to solve for.
    dt : float
        The time step size.

    Returns:
    u_hat : array_like
        The solution in Fourier space at each time step.
    """
    import numpy as np

    # Number of spatial points
    N = len(u0_hat)
    
    # Wavenumbers
    k = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    
    # Time steps
    num_steps = int(T / dt)
    
    # Initialize solution array
    u_hat = np.zeros((num_steps, N), dtype=complex)
    
    # Set initial condition
    u_hat[0, :] = u0_hat
    
    # Time-stepping loop
    if method != 'analytic':
        if method == 'leapfrog':
            #leapfrog method for the linear KdV equation in Fourier space
                for n in range(1, num_steps):
                    if n == 1:
                        u_hat[n, :] = u_hat[n-1, :] * np.exp(1j * k**3 * dt)
                    else:
                        u_hat[n, :] = u_hat[n-2, :] + 2*dt * (1j * k**3) * u_hat[n-1, :]

        if method == 'fe':
            # Forward Euler method for the linear KdV equation in Fourier space (unstable)
            for n in range(1, num_steps):
                u_hat[n, :] = (1 + dt * (-1j * k**3)) * u_hat[n-1, :]

    else:
        #analytic solution for the linear KdV equation in Fourier space
        for n in range(1, num_steps):
            # Update in Fourier space using the linear KdV equation
            u_hat[n, :] = u_hat[n-1, :] * np.exp(1j * k**3 * dt)
    return u_hat

