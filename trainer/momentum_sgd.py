"""
This class implements the 'Stochastic Gradient Descent' trainer.
"""

import numpy as np
import trainer
import sys
import a3c_literals as literals


class MomentumSGD(trainer.Trainer):
    """
    This class implements the 'Stochastic Gradient Descent' trainer.

    Calculation of the momentum:

            m := gamma * m + (1-alpha) * dJ(Theta)/dTheta

    Applying gradient:

            Theta = Theta - learning_rate * m

    """
    def __init__(self, shared_mem_name, gamma, learning_rate, num_parameters):
        """
        Initializer of the object.
        Args:
            shared_mem_name(string): Name of the globally shared memory.
            learning_rate(float): Learning rate for the SDG.
            gamma(float): velocity factor.
            num_parameters(int): Number of parameters i.e. length of the gradient.
        """
        self.gamma = gamma
        self.num_param = num_parameters
        self.momentum = np.zeros((num_parameters, 1))
        self.name = literals.MOMENTUM_SDG

        super(MomentumSGD, self).__init__(learning_rate=learning_rate, shared_mem_name=shared_mem_name)

    def perform_update(self, gradient):
        """
        This function performs an update on the globally shared memory.

        Args:
            gradient(array.py): Array containing gradient values. Size: [self.num_params, 1]
        Returns:

        """
        assert gradient.shape[0] == self.num_param
        assert gradient.shape[1] == 1

        self.calculate_momentum(gradient)
        w = sys.modules[self.shared_mem_name].__dict__["w"]
        w -= self.learning_rate * self.momentum

    def calculate_momentum(self, gradient):
        """
        This function updates the momentum vector following the formulas above.

        Args:
            gradient(array.py): Array containing gradient values. Size: [self.num_params, 1]
        Returns:

        """
        self.momentum = self.gamma * self.momentum + ( 1 - self.gamma ) * gradient
