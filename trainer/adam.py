"""
Class implements a trainer for Adam optimization.
"""
import numpy as np
import trainer
import sys
import a3c_literals as literals


class AdamOptimizer(trainer.Trainer):

    def __init__(self, shared_mem_name, size_w, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8):

        """Construct a new Adam optimizer.

            Initialization:

            m_0 <- 0 (Initialize initial 1st moment vector)
            v_0 <- 0 (Initialize initial 2nd moment vector)
            t <- 0 (Initialize time step)
            ```
            The update rule for `variable` with gradient `g` uses an optimization
            described at the end of section2 of the paper:
            ```
            t <- t + 1
            lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
            m_t <- beta1 * m_{t-1} + (1 - beta1) * g
            v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
            variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
        """

        self.name = literals.ADAM
        self.t = 0
        self.v_t = np.zeros((size_w, 1))
        self.m_t = np.zeros((size_w, 1))
        self.beta1 = b1
        self.beta2 = b2
        self.learning_rate_t = 0
        self.eps = e
        super(AdamOptimizer, self).__init__(learning_rate=learning_rate,
                                            shared_mem_name=shared_mem_name)

    def perform_update(self, gradient):
        """
        Executes update following the formulas from above.

        Args:
            gradient(array.py): Array containing the gradient values

        Returns:

        """

        self.t += 1

        # lr_t < - learning_rate * sqrt(1 - beta2 ^ t) / (1 - beta1 ^ t)
        self.learning_rate_t = self.learning_rate * np.sqrt(1 - np.power(self.beta2, self.t)) / np.sqrt(
            1 - np.power(self.beta1, self.t))

        # m_t < - beta1 * m_{t - 1} + (1 - beta1) * g
        self.m_t = np.multiply(self.beta1, self.m_t) + np.multiply((1-self.beta1), gradient)

        # v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
        self.v_t = np.multiply(self.beta2, self.v_t) + (1 - self.beta2) * np.multiply(gradient, gradient)

        # bias correction currently unused.
        # m_t_hat = m_t / (1 - beta1) ^ t
        # m_t_hat = np.divide(self.m_t, np.power(1 - self.beta1, self.t))

        # v_t_hat = v_t / (1 - beta2) ^ t
        # v_t_hat = np.divide(self.v_t, np.power(1 - self.beta2, self.t))

        w = sys.modules[self.shared_mem_name].__dict__["w"]
        w -= self.learning_rate_t * np.divide(self.m_t, np.sqrt(self.v_t) + self.eps)
