"""
Class implements a container to store all the necessary configurations to build workers.
"""
import trainer_configuration


class WorkerConfiguration(object):

    def __init__(self, trainer_config):
        """
        Initializes object.
        Args:
            trainer_config(trainer_configuration.TrainerConfiguration): Configuration of for the trainer.
        """
        self._worker_type = ''
        self._num_workers = 0
        self._environment_name = ''
        self._trainer_config = trainer_config
        self._shared_mem_name = ''
        self._seed = 0
        self._neural_network_name = ''
        self._state_size = 0
        self._action_size = 0
        self._num_episodes = 0
        self._episode_length = 0
        self._discount_gamma = 0
        self._entropy_factor = 1.0
        self._value_factor = 0.5
        self._steps_until_update = 5

    @property
    def num_workers(self):
        return self._num_workers

    @num_workers.setter
    def num_workers(self, num_workers):

        assert num_workers >= 0

        self._num_workers = num_workers

    @property
    def shared_mem_name(self):
        return self._shared_mem_name

    @shared_mem_name.setter
    def shared_mem_name(self, shared_mem_name):
        self._shared_mem_name = shared_mem_name

    @property
    def neural_network_name(self):
        return self._neural_network_name

    @neural_network_name.setter
    def neural_network_name(self, neural_network_name):
        self._neural_network_name = neural_network_name

    @property
    def environment_name(self):
        return self._environment_name

    @environment_name.setter
    def environment_name(self, environment_name):
        self._environment_name = environment_name

    @property
    def trainer_config(self):
        return self._trainer_config

    @trainer_config.setter
    def trainer_config(self, trainer_config):
        self._trainer_config = trainer_config

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed

    @property
    def action_size(self):
        return self._action_size

    @action_size.setter
    def action_size(self, action_size):

        assert action_size > 0

        self._action_size = action_size

    @property
    def state_size(self):
        return self._state_size

    @state_size.setter
    def state_size(self, state_size):

        assert state_size > 0

        self._state_size = state_size

    @property
    def worker_type(self):
        return self._worker_type

    @worker_type.setter
    def worker_type(self, worker_type):
        self._worker_type = worker_type

    @property
    def num_episodes(self):
        return self._num_episodes

    @num_episodes.setter
    def num_episodes(self, num_episodes):

        assert num_episodes > 0

        self._num_episodes = num_episodes

    @property
    def episode_length(self):
        return self._episode_length

    @episode_length.setter
    def episode_length(self, acts_per_episode):

        assert acts_per_episode > 0

        self._episode_length = acts_per_episode

    @property
    def discount_gamma(self):
        return self._discount_gamma

    @discount_gamma.setter
    def discount_gamma(self, discount_gamma):

        assert discount_gamma >= 0
        assert discount_gamma <= 1

        self._discount_gamma = discount_gamma

    @property
    def entropy_factor(self):
        return self._entropy_factor

    @entropy_factor.setter
    def entropy_factor(self, entropy_factor):
        self._entropy_factor = entropy_factor

    @property
    def value_factor(self):
        return self._value_factor

    @value_factor.setter
    def value_factor(self, value_factor):
        self._value_factor = value_factor

    @property
    def steps_until_update(self):
        return self._steps_until_update

    @steps_until_update.setter
    def steps_until_update(self, steps_until_update):
        self._steps_until_update = steps_until_update





