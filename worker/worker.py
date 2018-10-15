"""
This class implements a base class for the worker.
"""

import copy
import sys

import numpy as np

from environment.environment import BaseEnvironment
from neural_net.neural_net_base import BaseModel
from trainer.trainer import Trainer


class Worker(object):

    def __init__(self, neural_network,
                 environment,
                 trainer,
                 shared_mem_name,
                 worker_id,
                 seed,
                 num_episodes,
                 episode_length,
                 discount_gamma,
                 steps_until_update):

        """
        Initializer of the worker class.

        Args:
            neural_network(BaseModel): The neural network.
            environment(BaseEnvironment): The environment
            trainer(Trainer): The trainer executing the gradient updates.
            shared_mem_name(string): Name of the temporary shared memory module
            worker_id(int): Id of the worker.
            seed(int): Seed for random number generations
            num_episodes(int): total number of episodes
            episode_length(int): Maximal number of actions to perform in one episode
            discount_gamma(float): Ranging from zero to one. discount-factor for rewards.
            steps_until_update(int): How many steps to perform in the environment before updating the parameters

        """

        self.neural_network = neural_network
        self.shared_mem_name = shared_mem_name
        self.name = 'worker_' + str(worker_id)
        self.environment = environment
        self.trainer = trainer
        self.random = np.random.RandomState(seed=seed)
        self.T_max = num_episodes
        self.t_max = episode_length
        self.discount_gamma = discount_gamma
        self.steps_until_update = steps_until_update

    def get_global_parameters(self):
        """
        This function accesses the global shared memory and creates a shallow local copy of the memory.

        Returns:
            w(array.py): Shallow copy of the current global neural network parameters

        """

        w = copy.copy(sys.modules[self.shared_mem_name].__dict__["w"])
        return w

    def update_local_weights(self, op, weights, sess):

        """
        This function updates the local weights. The set_weights function from the BaseNeuralNet class provides
        a function (set_weights()) that returns the parameters for this function.
        Args:
            op(List of tensors): Tensorflow operations for assigning values to the weights of the network.
            weights(tf placeholder): Placeholder for the weights.
            sess(Tensorflow session): A tensorflow session.

        Returns:

        """

        sess.run(op, feed_dict={weights: self.get_global_parameters()})

    def work(self):

        raise NotImplementedError
