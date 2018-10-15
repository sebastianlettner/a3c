import sys
import time
from collections import namedtuple

import numpy as np
import tensorflow as tf

import worker
from environment.environment import BaseEnvironment
from neural_net.neural_net_base import BaseModel
from trainer.trainer import Trainer

Step = namedtuple('Step', 'cur_step action next_step reward done')


class A3CWorker(worker.Worker):

    def __init__(self,
                 neural_network,
                 environment,
                 trainer,
                 shared_mem_name,
                 worker_id,
                 seed,
                 num_episodes,
                 episode_length,
                 steps_until_update,
                 discount_gamma):

        """
        Initializer of the worker class.

        Args:
            neural_network(BaseModel): The neural network.
            environment(BaseEnvironment): The environment.
            trainer(Trainer): The trainer executing the gradient updates.
            shared_mem_name(string): Name of the temporary shared memory module
            worker_id(int): Id of the worker.
            seed(int): Seed for random number generations
            num_episodes(int): total number of episodes
            episode_length(int): Maximal number of actions to perform in one episode
            discount_gamma(float): Ranging from zero to one. discount-factor for rewards.
            steps_until_update(int): How many steps to perform in the environment before updating the parameters


        """

        super(A3CWorker, self).__init__(neural_network=neural_network,
                                        environment=environment,
                                        trainer=trainer,
                                        shared_mem_name=shared_mem_name,
                                        worker_id=worker_id,
                                        seed=seed,
                                        num_episodes=num_episodes,
                                        episode_length=episode_length,
                                        discount_gamma=discount_gamma,
                                        steps_until_update=steps_until_update)

        # preallocate memory for gradients
        self.gradient = np.zeros(dtype=np.float32, shape=[self.neural_network.get_total_num_weights(), 1])

    def apply_gradients_global(self):
        """
        This function pushes the gathered experience of an episode to the global parameter memory using the update
        rule of the specified trainer.

        Returns:

        """
        self.trainer.perform_update(self.gradient)

    def work(self):

        """
        This function executes the asynchronous advantage actor critic algorithm from the paper:
        "Asynchronous methods for Deep Reinforcement learning".


        Returns:
            sess(tensorflow session): A tensorflow session.

        """
        t_start = time.time()

        # create the tf session.
        sess = tf.Session()

        assert self.T_max > 0
        assert self.t_max > 0
        assert self.discount_gamma <= 1
        assert self.discount_gamma > 0

        global_update_count = 0

        # initialize all variables.
        sess.run(tf.global_variables_initializer())

        # get the current/starting state from the environment
        cur_state = self.environment.reset()
        episode_len = 1

        cum_reward = 0

        total_reward = 0

        # get the tensors and the placeholder necessary for updating local weights.
        op, weights_global = self.neural_network.set_weights()

        init_flag = 1

        while global_update_count < self.T_max:

            # Before the first episode is played the first worker set his weights globally.
            # Otherwise the global weights would be all zero.
            if init_flag and self.name == 'worker_0':
                w = sys.modules[self.shared_mem_name].__dict__["w"]
                weights = self.neural_network.get_weights(sess)
                for i in range(self.neural_network.get_total_num_weights()):
                    w[i][0] = weights[i][0]
                init_flag = 0

            # Synchronize local parameters with global parameters.
            else:
                self.update_local_weights(op, weights_global, sess)

            steps = []

            # play an episode of length episode_length (if not terminal before).
            for i in xrange(self.steps_until_update):
                # Get prediction for current state.
                action = self.neural_network.get_action(cur_state, sess)

                # Execute the action in the environment. Get the next_state the reward and the terminal flag.
                next_state, reward, terminal, _ = self.environment.step(action)
                cum_reward += reward
                episode_len += 1

                # save the information received from the environment.
                steps.append(Step(cur_step=cur_state,
                                  action=action,
                                  next_step=next_state,
                                  reward=reward,
                                  done=terminal))

                # if the state was terminal or the episode reached a length of over 10000 the episode is considered
                # finished.
                if terminal or episode_len >= self.t_max:
                    # reset the environment
                    cur_state = self.environment.reset()
                    print '{}: episode {} finished in {} steps, cumulative reward: {}'.format(self.name,
                                                                                              global_update_count,
                                                                                              episode_len,
                                                                                              cum_reward)
                    total_reward += cum_reward
                    cum_reward = 0
                    global_update_count += 1
                    episode_len = 0
                    break

                cur_state = next_state

            # if t_max was reached or the episode was terminal perform update...

            # if the last state was terminal the reward is zero.
            if steps[-1].done:
                R = 0
            # else we bootstrap form the neural network.
            else:
                R = self.neural_network.predict_value(cur_state, sess)

            # get the gradients w.r.t. the gathered experience.
            grads = self.train(steps, sess, R)

            # set the workers gradients to calculated above.
            self.set_gradients(grads)

            # perform update on the global weights.
            self.apply_gradients_global()

        print "Time for " + str(global_update_count) + " eps: " + str(time.time() - t_start) + " in " + self.name
        return sess

    def train(self, steps, sess, R):

        """
        This functions calculates the gradients.

        Args:
            steps(list): List containing the experience.
            sess(tensorflow session): A Tensorflow session.
            R(float): Bootstrapped value if last state was not terminal. Zero otherwise.

        Returns:
            grads(List of numpy arrays): The numerical gradients for every weight in the network.

        """

        # make list of lists to numpy array of size (len(outer list), len(list_element))
        r_batch = np.zeros(len(steps))
        for i in reversed(xrange(len(steps))):
            step = steps[i]
            R = step.reward + self.discount_gamma * R
            r_batch[i] = R

        cur_state_batch = [step.cur_step for step in steps]

        pred_v_batch = self.neural_network.predict_value(cur_state_batch, sess)

        action_batch = [step.action for step in steps]

        advantage_batch = [r_batch[i] - pred_v_batch[i] for i in xrange(len(steps))]

        action_batch = np.reshape(action_batch, [-1])

        advantage_batch = np.reshape(advantage_batch, [-1])

        r_batch = np.reshape(r_batch, [-1])

        feed_dict = {
            self.neural_network.input_s: cur_state_batch,
            self.neural_network.input_a: action_batch,
            self.neural_network.advantage: advantage_batch,
            self.neural_network.target_v: r_batch,
        }

        return sess.run(self.neural_network.gradients, feed_dict=feed_dict)

    def set_gradients(self, gradients):

        """
        This function takes the tensorflow gradient and converts them to a flat numpy array.
        Args:
            gradients(list): list of numpy arrays containing the numerical values of the gradients w.r.t the weights

        Returns:

        """

        offset = 0
        for g in gradients:
            try:
                g_dims = g.shape[0] * g.shape[1]
            except IndexError:
                g_dims = g.shape[0]
            g = g.reshape(g_dims, 1)
            # set the gradients
            self.gradient[offset:offset + g_dims][:] = g
            offset += g_dims


