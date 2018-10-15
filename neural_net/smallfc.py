import tensorflow as tf
import tensorflow.contrib.slim as slim

from neural_net.neural_net_base import BaseModel


class SmallFC(BaseModel):
    """
    This class implements a small fully connected neural network with two hidden layers and output layer according to
    the a3c algorithm.
    """
    def __init__(self,
                 state_size,
                 action_size,
                 name,
                 num_n1=256,
                 num_n2=256,
                 entropy_factor=1.0,
                 value_factor=0.5):
        """

        Args:
            state_size(int): Size of the state vector.
            action_size(int): Size of the action vector.
            name(str): Name for the variable scope.
            num_n1(int): Number of neurons in the first layer.
            num_n2(int): Number of neurons in the second layer.
            entropy_factor(float): Weight you want to give the entropy loss.
                                   Values between zero and one inclusive worked well.
        """
        self.num_n1 = num_n1
        self.num_n2 = num_n2
        super(SmallFC, self).__init__(state_size, action_size, name, entropy_factor, value_factor)

    def build_network(self, name):

        input_s = tf.placeholder(tf.float32, [None, self.state_size], name='state')
        input_a = tf.placeholder(tf.int32, [None], name='action')
        advantage = tf.placeholder(tf.float32, [None], name='adv')
        target_v = tf.placeholder(tf.float32, [None], name='target_v')

        with tf.variable_scope(name):

            l1 = slim.fully_connected(input_s, self.num_n1, activation_fn=tf.nn.relu)
            l2 = slim.fully_connected(l1, self.num_n2, activation_fn=tf.nn.relu)

            policy = slim.fully_connected(l2, self.action_size, activation_fn=tf.nn.softmax)
            value = slim.fully_connected(l2, 1, activation_fn=tf.nn.relu)

        action_mask = tf.one_hot(input_a, self.action_size, 1.0, 0.0)
        action_est = tf.reduce_sum(policy * action_mask, 1)

        model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return input_s, input_a, advantage, target_v, policy, value, action_est, model_variables


