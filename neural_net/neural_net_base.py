import tensorflow as tf
import numpy as np


class BaseModel(object):
    """
    This class implements a base model for neural networks fitting the a3c algorithm.
    """
    def __init__(self, state_size, action_size, name, entropy_factor=1.0, value_factor=0.5):
        """
        Initializes object.
        Args:
            state_size(int): Size of the state vector.
            action_size(int): Size of the action vector.
            name(str): Name of the variable scope.
            entropy_factor(float): Contribution of the entropy loss to the total loss.
            value_factor(float): Contribution of the value loss to the total loss.

        """
        self.state_size = state_size
        self.action_size = action_size
        self.name = name
        self.entropy_factor = entropy_factor
        self.value_factor = value_factor

        self.input_s, self.input_a, self.advantage, self.target_v, self.policy, self.value, self.action_est, \
            self.model_variables = self.build_network(name)

        self.value_loss = self.value_factor * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
        self.entropy_loss = 1.0 * tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10))
        self.policy_loss = 1.0 * tf.reduce_sum(-tf.log(self.action_est + 1e-10) * self.advantage)
        self.loss = self.value_loss + self.policy_loss + (self.entropy_factor * self.entropy_loss)
        self.gradients = tf.gradients(self.loss, self.model_variables)
        # self.gradients , _ = tf.clip_by_global_norm(self.gradients, 40.0)

    def build_network(self, name):
        """
        This function implements the architecture of the network.
        Args:
            name(str): Name of the tensorflow variable scope.

        Returns:
            input_state(placeholder): Placeholder for the input. Takes variables of size [None, state_size]
            input_action(placeholder): Placeholder for the actions. This will be plugged into
                                  the graph for gradient calculation. Takes variables of
                                  size [None]. Actions are scalars.
            advantages(placeholder): Placeholder for the advantages. This will be plugged in
                                     the graph for gradient calculation. Takes variables of size [None]
                                     Advantages are scalars.
            target_values(placeholder): Placeholder for the target values that the value function
                                        should approximate. This will be plugged in into the gradient
                                        calculation. It takes variables of size [None]
                                        The target values are scalars.
            policy(tensor): Output-layer with num_neurons = action_size and softmax activation.
            value(tensor): Output-layer with num_neurons = 1 and linear activation.
            action_estimation(tensor): Tensor calculating the probability of action a_t in state s_t.
            model_variables(tensor): Collection of TRAINING_VARIABLES.


        """
        raise NotImplementedError

    def get_action(self, state, sess):
        """
        This function returns an action given a state w.r.t. the current policy.
        The action is sampled using numpy's random.choice function.
        Args:
            state(array.py): State vector. The vector needs to support a reshape operation to [1, state_size].
            sess(tensorflow session): A tensorflow session.

        Returns:
            action(int): Sampled action.
        """
        state = np.reshape(state, [-1, self.state_size])
        policy = sess.run(self.policy, feed_dict={self.input_s: state})
        return np.random.choice(range(self.action_size), p=policy[0])

    def predict_policy(self, state, sess):
        """
        This function returns the policy of a given state w.r.t. the current policy.
        Args:
            state(array.py): State vector. The vector needs to support a reshape operation to [1, state_size].
            sess(tensorflow session): A tensorflow session.

        Returns:
            policy(array.py): Array containing the portability of each state. Size: [1, action_size].
        """
        state = np.reshape(state, [-1, self.state_size])
        policy = sess.run(self.policy, feed_dict={self.input_s: state})
        return policy[0]

    def predict_value(self, state, sess):
        """
        This function returns the value of a given state w.r.t. the current value estimation.
        Args:
            state(array.py): State vector. The vector needs to support a reshape operation to [1, state_size].
            sess(tensorflow session): A tensorflow session.

        Returns:
            value(float): Value of the state.
        """
        state = np.reshape(state, [-1, self.state_size])
        return sess.run(self.value, feed_dict={self.input_s: state})

    def get_total_num_weights(self):
        """
        This function returns the total number of parameters in the network.
        Returns:
            num_parameters(int): Number of network parameters.

        """
        return np.sum([np.prod(v.get_shape().as_list()) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                   self.name)])

    def set_weights(self):
        """
        This function implements a tensorflow graph to set all network weights.
        Returns:
            ops(list): List of tensorflow operations to set all network weights.
            weights(placeholder): Placeholder for the parameters of size [total_num_parameters, 1]

        """
        global_weights = tf.placeholder(dtype=tf.float32, shape=[self.get_total_num_weights(), 1])
        assert np.shape(global_weights)[0] == self.get_total_num_weights()

        local_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        offset = 0
        op_holder = []
        for weight in local_weights:
            try:
                weight_dims = weight.shape[0] * weight.shape[1]
            except IndexError:
                weight_dims = weight.shape[0]
            weight_dims = int(weight_dims)
            params = global_weights[offset:offset+weight_dims][:]
            params = tf.reshape(params, weight.shape)
            offset += weight_dims
            op_holder.append(weight.assign(params))

        return op_holder, global_weights

    def get_weights(self, sess):

        """
        Function returns all network weights in a flat numpy array.
        Args:
            sess(tensorflow session): A tensorflow session.
        Returns:
            parameters(array.py): Numpy array of size [total_num_parameters, 1] containing all weight values.

        """

        # get the tensors from the local net using the scope
        local_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

        # evaluate the values
        values = sess.run(local_weights)

        # convert the values to a flat numpy array
        parameters = np.zeros(shape=[self.get_total_num_weights(), 1])
        offset = 0
        for val in values:
            try:
                val_dims = val.shape[0] * val.shape[1]

            except IndexError:
                val_dims = val.shape[0]
            val = val.reshape(val_dims, 1)
            parameters[offset:offset+val_dims][:] = val
            offset += val_dims

        return parameters

    def print_weights(self, sess):
        """
        This function prints all the weight values in the network.
        Args:
            sess:

        Returns:

        """
        print sess.run(self.model_variables)



