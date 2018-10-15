"""
Implementing interface for trainers.
"""


class Trainer(object):

    def __init__(self, learning_rate, shared_mem_name):
        """
        Initializes object.
        Args:
            learning_rate(float): Learning rate of the trainer
            shared_mem_name(string): Name of the globally shared memory
        """

        self._learning_rate = learning_rate
        self._shared_mem_name = shared_mem_name
        self._name = None

    def perform_update(self, gradient):
        """
        Implemented in subclasses.
        Args:
            gradient(array.py): Gradient values.

        Returns:

        """
        raise NotImplementedError

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def shared_mem_name(self):
        return self._shared_mem_name

    @shared_mem_name.setter
    def shared_mem_name(self, shared_mem_name):
        self._shared_mem_name = shared_mem_name

