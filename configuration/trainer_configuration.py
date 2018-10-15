"""
Class implements a container for the configuration data of a trainer.
"""
import a3c_literals as literal


class TrainerConfiguration(object):
    """
    Base Class for Configurations.
    """
    def __init__(self):
        """
        Initializes object.
        """
        self._learning_rate = 0.001
        self._name = None

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


class MomentumSDGConfig(TrainerConfiguration):

    def __init__(self):
        """
        Initializes object.
        """
        self._name = literal.MOMENTUM_SDG
        self._gamma = 0.999
        super(MomentumSDGConfig, self).__init__()

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma


class GradientDescentConfig(TrainerConfiguration):

    def __init__(self):

        """
        Initializes object
        """

        self._name = literal.GRADIENT_DESCENT
        super(GradientDescentConfig, self).__init__()


class AdamConfig(TrainerConfiguration):

    def __init__(self):

        self._name = literal.ADAM
        self.b1 = 0.9
        self.b2 = 0.999
        self.eps = 1e-8

        super(AdamConfig, self).__init__()


