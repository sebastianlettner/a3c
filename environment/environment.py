"""
Class implementing the interface for environments.
"""


class BaseEnvironment(object):

    def __init__(self):
        pass
        # self.current_state = None

    def get_action_size(self):
        """
        This function return the total number of possible actions.

        Returns:
            total_num_actions(int): Dimension of the action space as scalar

        """
        raise NotImplementedError

    def get_state_size(self):
        """
        This function returns the dimensions of the state space

        Returns:
            state_space_dims(list): Dimension of the state space e.g. [255, 255, 3] or [100000, 1]

        """
        raise NotImplementedError

    def get_start_state(self):
        """
        This function needs to return the initial state.
        Every episode will started at this state. The state needs to be of shape self.get_state_size().

        Returns:
            initial_state(numpy array): Numpy array encoding the initial state with size self.get_state_size()

        """
        raise NotImplementedError

    def step(self, action):
        """
        This function perform the given action on the current state. It then updates the current state
        and returns the reward.

        Args:
            action(numpy array): Encoding of the action.

        Returns:
            reward(float): Reward of the state after transmission.

        """
        raise NotImplementedError
        # return self.current_state, reward

    def is_episode_finished(self):
        """
        This function provides information about whether a state is terminal or not.

        Returns:
            is_episode_finsihed(bool): True: --> current state is terminal
                                       False: --> current state is not terminal
        """

    def reset(self):
        """
        This function resets the environment to start a new episode.
        Returns:

        """
        raise NotImplementedError


