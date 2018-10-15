"""
This class implements the factory pattern for NeuralNetworks.
"""
import a3c_literals as literals
from neural_net.smallfc import SmallFC


class NeuralNetworkFactory(object):

    @classmethod
    def produce(cls, network_name, a_size, s_size, scope_name, entropy_factor, value_factor):
        """
        This function produces neural network instances.
        Args:
            network_name(string): Name of the Neural Net.
            a_size(int): Size of the action vector.
            s_size(int): Size of the state vector.
            entropy_factor(float): Contribution of the entropy loss to the total loss.
            value_factor(float): Contribution of the value loss to the total loss.

        Returns:
            neural_network(): Concrete neural network.

        """
        if network_name == literals.SMALL_FC_NET:
            return SmallFC(action_size=a_size,
                           state_size=s_size,
                           name=scope_name,
                           entropy_factor=entropy_factor,
                           value_factor=value_factor)

        else:
            raise Exception("Unknown neural network name: " + str(network_name))