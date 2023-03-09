"""
Pendulum with disturbance only on control
"""

import autokoopman.core.system as asys
import sympy as sp
import numpy as np

class InvertedPendulumWithInputAndDisturbcontrol(asys.SymbolicContinuousSystem):
    r"""
    Simple Pendulum with Constant Torque Input
        We model a pendulum with the equation:

        .. math::
            l^2 \ddot{\theta} + \beta \theta + g l \sin \theta = \tau

        This leads to the state space form:
        
        .. math::
            \left\{\begin{array}{l}
            \dot{x}_{1} = x_2 \\
            \dot{x}_{2} = - g / l \sin x_1 - 2 \beta x_2 + u_1 \\
            \end{array}\right.

        Note that :math:`\beta=b / m` kn this formulation: http://underactuated.mit.edu/pend.html .
    """
    # NOTE: difference here is that gravity is flipped
    def __init__(self, g=-1*9.81, l=1.0, beta=0.0, num_timepoints=10, samp_period=0.1):
        self.name = "inverted_pendulum_withD_oncontrol"
        # self.init_set_low = [-1, -1]
        # self.init_set_high = [1, 1]

        self.init_set_low = [-0.5, -0.1]
        self.init_set_high = [0.5, 0.1]

        self.input_type = ["step", "rand"]
        # self.teval = np.linspace(0, 10, 200)
        # self.teval = np.linspace(0, 10, num_timepoints)
        self.teval = np.arange(num_timepoints) * samp_period
        
        self.input_set_low = [-1]
        self.input_set_high = [1]

        # disturbance as input
        self.disturb_proportion = 0.1
        self.input_set_low.append(self.input_set_low[0] * self.disturb_proportion)
        self.input_set_high.append(self.input_set_high[0] * self.disturb_proportion) 

        print("inverted pendulum with D on control: " + str(self.input_set_low) + " : " + str(self.input_set_high))
        print("WITH Disturbance")
        
        # variables 
        theta, thetadot = sp.symbols("theta thetadot")
        tau = sp.symbols("tau")
        disturb = sp.symbols("disturb") # disturbance to add to control

        xdot = [thetadot, -g / l * sp.sin(theta) - 2 * beta * thetadot + tau + disturb]
        super(InvertedPendulumWithInputAndDisturbcontrol, self).__init__(
            (theta, thetadot), xdot, input_variables=(tau, disturb)
        )