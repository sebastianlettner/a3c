"""
Class implements the factory pattern for trainers.
"""
from trainer.momentum_sgd import MomentumSGD
from trainer.gradient_descent import GradientDescent
from trainer.adam import AdamOptimizer
import a3c_literals as literals
from trainer.trainer import Trainer


class TrainerFactory(object):

    @classmethod
    def produce(cls, trainer_config, shared_mem_name, num_parameters):
        """

        Args:
            trainer_config(): Configuration for the trainer.
            shared_mem_name(str): Name of the shared memory module.
            num_parameters(int): Number of parameters of the neural net.

        Returns:
            trainer(Trainer): Concrete trainer.

        """
        if trainer_config.name is None:
            raise Exception("No name for trainer configured.")

        elif trainer_config.name == literals.MOMENTUM_SDG:

            return MomentumSGD(shared_mem_name=shared_mem_name,
                               gamma=trainer_config.gamma,
                               learning_rate=trainer_config.learning_rate,
                               num_parameters=num_parameters)
        elif trainer_config.name == literals.GRADIENT_DESCENT:

            return GradientDescent(shared_mem_name=shared_mem_name,
                                   learning_rate=trainer_config.learning_rate)

        elif trainer_config.name == literals.ADAM:

            return AdamOptimizer(shared_mem_name=shared_mem_name,
                                 size_w=num_parameters,
                                 learning_rate=trainer_config.learning_rate,
                                 b1=trainer_config.b1,
                                 b2=trainer_config.b2,
                                 e=trainer_config.eps)

        else:
            raise ValueError("Unknown Trainer name: " + str(trainer_config.name))
