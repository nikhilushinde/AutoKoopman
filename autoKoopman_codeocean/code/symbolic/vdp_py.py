"""
Van Der Pol System with control and disturbance.
"""

import autokoopman.core.system as asys
import sympy as sp
import numpy as np

class VanderPol(asys.SymbolicContinuousSystem):
    r"""
    Van der Pol

    .. math:
        \dot x = \begin{bmatrix} x_2 \\ 
        \mu(1-x_1^2)x_2 - \gamma x_1\end{bmatrix} + 
        \begin{bmatrix} 0 \\ 
        b\end{bmatrix} u + 
        C d

    The scalar :math:`\mu` is a parameter indicating the nonlinearity and the strength of the damping. 
    The parameter :math:`\gamma` is often = 1.

    """

    def __init__(self, mu=1., gamma=1., disturbance_on_all=True):
        self.name = "VanderPol"
        self.init_set_low = [-4 for i in range(2)]
        self.init_set_high = [4 for i in range(2)]
        self.input_type = ["step", "rand"]
        self.teval = np.linspace(0, 10, 200)
        
        self.input_set_low = [-1 for i in range(1)]
        self.input_set_high = [1 for i in range(1)]

        # disturbance as input
        self.disturb_proportion = 0.5

        print("Van der Pol, input bounds:" + str(self.input_set_low) + " : " + str(self.input_set_high))

        # variables 
        x1, x1dot = sp.symbols("x1 x1dot")
        u = sp.symbols("u")

        if disturbance_on_all:
            self.input_type.append("rand")

            print("includes disturbance on all")

            self.input_set_low.append(self.input_set_low[0] * self.disturb_proportion)
            self.input_set_high.append(self.input_set_high[0] * self.disturb_proportion) 
            self.input_set_low.append(self.input_set_low[0] * self.disturb_proportion)
            self.input_set_high.append(self.input_set_high[0] * self.disturb_proportion) 

            d1, d2 = sp.symbols("d1 d2")

            xdot = [x1dot + d1, 
                mu * (1 - x1 ** 2) * x1dot - gamma * x1 + u + d2]
            
            super(VanderPol, self).__init__((x1, x1dot), xdot, input_variables=(u, d1, d2))
        
        else:

            print("includes disturbance on ctrl")

            self.input_set_low.append(self.input_set_low[0] * self.disturb_proportion)
            self.input_set_high.append(self.input_set_high[0] * self.disturb_proportion) 

            d = sp.symbols("d")

            xdot = [x1dot, 
                mu * (1 - x1 ** 2) * x1dot - gamma * x1 + u + d]
            
            super(VanderPol, self).__init__((x1, x1dot), xdot, input_variables=(u, d))