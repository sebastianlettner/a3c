"""
Class implements the factory pattern for Environments.
"""
import a3c_literals as literals
import sys
# sys.path.append("...")
import gym


class EnvironmentFactory(object):

    @classmethod
    def produce(cls, environment_name):
        """
        This function produces environment instances.
        Args:
            environment_name(string): Name of the environment.

        Returns:
            environment(BaseEnvironment).

        """

        if environment_name == literals.CART_POLE:
            return gym.make('CartPole-v0')

        else:
            raise Exception('Unknown Environment name' + str(environment_name))

