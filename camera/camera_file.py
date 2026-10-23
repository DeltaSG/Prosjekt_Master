# Creating a camera class, that emulates the physical setup as described in Rediscovering the shallow-water equations

import numpy as np 

class Camera():
    def __init__(self, L, N):
        """
        Fixing the physical parameters of the set-up.
        Parameters
        L: Length of the camera view.
        N: Number of pixels in "x" direction of the camera
        """
        self.L = L
        self.N = N
        self.dx = L/N

        #array used for plotting
        self.x = np.linspace(-L/2, L/2 , N)