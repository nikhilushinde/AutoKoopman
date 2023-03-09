"""
Duffing System with control and disturbance.
"""

import autokoopman.core.system as asys
import sympy as sp
import numpy as np

class Duffing(asys.SymbolicContinuousSystem):
    r"""
    Duffing

    .. math:
        \dot x = \begin{bmatrix} x_2 \\ 
        \mu(1-x_1^2)x_2 - \gamma x_1\end{bmatrix} + 
        \begin{bmatrix} 0 \\ 
        b\end{bmatrix} u + 
        C d

    where :math`\alpha` controls linear stiffness, :math`\delta` controls damping (often small or 0), 
    and :math`\beta` controls the amount of nonlinearity. Generally assume all constants are positive for 
    oscillatory behavior. 

    """

    def __init__(self, alpha=1., beta=1., delta=0.1, disturbance_on_all=True):
        self.name = "Duffing"
        self.init_set_low = [-1.75 for i in range(2)]
        self.init_set_high = [1.75 for i in range(2)]
        self.input_type = ["step", "rand"]
        self.teval = np.linspace(0, 10, 200)
        
        self.input_set_low = [-1 for i in range(1)]
        self.input_set_high = [1 for i in range(1)]

        self.disturb_proportion = 0.5

        print("Duffing, input bounds:" + str(self.input_set_low) + " : " + str(self.input_set_high))

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
                 alpha * x1 - beta * x1 ** 3 - delta * x1dot + u + d2]
            
            super(Duffing, self).__init__((x1, x1dot), xdot, input_variables=(u, d1, d2))

        else:

            print("includes disturbance on ctrl")

            self.input_set_low.append(self.input_set_low[0] * self.disturb_proportion)
            self.input_set_high.append(self.input_set_high[0] * self.disturb_proportion) 

            d = sp.symbols("d")

            xdot = [x1dot, 
                 alpha * x1 - beta * x1 ** 3 - delta * x1dot + u + d]
            
            super(Duffing, self).__init__(
            (x1, x1dot), xdot, input_variables=(u, d))