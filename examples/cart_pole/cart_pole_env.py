from configuration.worker_configuration import WorkerConfiguration
from configuration.trainer_configuration import AdamConfig
import a3c_literals as literals
import time
from multiprocessing import Process
from shared_model.shared_memory import SharedMemory
from factories.worker_factory import WorkerFactory

STATE_SIZE = 4
ACTION_SIZE = 2
LEARNING_RATE = 0.001

DISCOUNT_GAMMA = 0.999
EPSILON = 0.9
EPISODE_LENGTH = 200
NUM_EPISODES = 750

if __name__ == '__main__':

    trainer_config = AdamConfig()
    trainer_config.name = literals.ADAM
    trainer_config.learning_rate = LEARNING_RATE

    worker_config = WorkerConfiguration(trainer_config)
    worker_config.discount_gamma = DISCOUNT_GAMMA
    worker_config.epsilon = EPSILON
    worker_config.episode_length = EPISODE_LENGTH
    worker_config.num_episodes = NUM_EPISODES
    worker_config.state_size = 4
    worker_config.action_size = 2
    worker_config.num_workers = 8  # mp.cpu_count()
    worker_config.shared_mem_name = '_shared_model_'
    worker_config.seed = 3
    worker_config.neural_network_name = literals.SMALL_FC_NET
    worker_config.environment_name = literals.CART_POLE
    worker_config.worker_type = literals.A3C_WORKER
    worker_config.steps_until_update = 5
    worker_config.value_factor = 0.5
    worker_config.entropy_factor = 1.0

    workers = WorkerFactory.produce(worker_config)
    with SharedMemory(workers[0].neural_network.get_total_num_weights()) as sm:
        processes = []
        for worker in workers:
            processes.append(Process(target=worker.work, args=[]))

        for p in processes:
            p.start()
            time.sleep(0.1)

        for p in processes:
            p.join()
