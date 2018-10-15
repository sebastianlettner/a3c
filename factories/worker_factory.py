"""
This class implements the factory pattern to create workers.
"""
import a3c_literals as literals
from configuration.worker_configuration import WorkerConfiguration
from environment_factory import EnvironmentFactory
from network_factory import NeuralNetworkFactory
from trainer_factory import TrainerFactory
from worker.a3c_worker import A3CWorker


class WorkerFactory(object):

    @classmethod
    def produce(cls, worker_config):
        """
        This function produces worker instances.
        Args:
            worker_config(WorkerConfiguration): Configuration for worker.

        Returns:
            workers(list): List of workers ready to work.

        """
        workers = []

        for i in range(worker_config.num_workers):

            # create environment
            env = EnvironmentFactory.produce(environment_name=worker_config.environment_name)

            # create neural network
            net = NeuralNetworkFactory.produce(network_name=worker_config.neural_network_name,
                                               a_size=worker_config.action_size,
                                               s_size=worker_config.state_size,
                                               scope_name="worker_" + str(i),
                                               entropy_factor=worker_config.entropy_factor,
                                               value_factor=worker_config.value_factor)

            # create trainer
            trainer = TrainerFactory.produce(trainer_config=worker_config.trainer_config,
                                             shared_mem_name=worker_config.shared_mem_name,
                                             num_parameters=net.get_total_num_weights())

            # create worker
            if worker_config.worker_type == literals.A3C_WORKER:
                worker = A3CWorker(neural_network=net,
                                   environment=env,
                                   trainer=trainer,
                                   shared_mem_name=worker_config.shared_mem_name,
                                   worker_id=i,
                                   seed=worker_config.seed,
                                   num_episodes=worker_config.num_episodes,
                                   episode_length=worker_config.episode_length,
                                   discount_gamma=worker_config.discount_gamma,
                                   steps_until_update=worker_config.steps_until_update)
                workers.append(worker)

            else:
                raise Exception('Unknown worker type: ' + str(worker_config.worker_type))

        return workers



