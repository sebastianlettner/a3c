"""
Class implements a trainer for gradient descent optimization.
"""
import numpy as np
import trainer
import sys
import a3c_literals as literals


class GradientDescent(trainer.Trainer):

    def __init__(self, shared_mem_name, learning_rate):
        """
        Initializer of the object.
        Args:
            shared_mem_name(string): Name of the globally shared memory.
            learning_rate(float): Learning rate for the GD optimizer.
        """
        self.name = literals.GRADIENT_DESCENT
        super(GradientDescent, self).__init__(learning_rate=learning_rate,
                                              shared_mem_name=shared_mem_name)

    def perform_update(self, gradient):
        """
        This function performs one gradient descent update step.
            w = w - learning_rate * dw
        Args:
            gradient(array): Array containing the gradient values.

        Returns:

        """
        w = sys.modules[self.shared_mem_name].__dict__["w"]
        w -= self.learning_rate * gradient
