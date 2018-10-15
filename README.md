<header> 
<h1> 
Efficient Reinforcement Learing - Asynchronous Advantage Actor Critic
</h1>
<h2>
Structure
</h2>
</header>

- configuration: Container classes storing all parameters.
- environment: Contains interface class for concrete environments.
- factories: Factories to creating set ups.
- helpers: Some helpers.
- neural_net: Contains interface class for concrete neural networks
- shared_model: Shared memory stuff.
- tests: Example setups.
- trainer: Optimizers for training.
- worker: A3C-Worker running the algorithm.

<header>
<h2>
Requirements
</h2>
</header>

- Numpy (http://www.numpy.org/)
- Tensorflow (https://www.tensorflow.org/)
- Scipy (https://www.scipy.org/)

<header>
<h2>
Usage
</h2>

In order to use the framework the
<ul type 'square'>
<li>environment</li>
<li>neural network architecture</li>
</ul>
has to be implemented from your side.<br></br>
<br></br>
To do so use the interface for neural networks and the environment.<br></br>
<br></br>
For the neural network you only have to overwrite the <i>build_network</i> function
located in the <b>BaseModel</b> class in the file neural_nets/neural_net_base.py<br></br>
Implement this function as specified in the <b>BaseModel</b> class.<br></br>
A small FF Neural Network is implemented as an example. <br></br>
<br></br>
Also the <b>BaseEnvironment</b> class in environment/environment.py needs to
be implemented as specified.<br></br>
Also have a look at the configuration/worker_configurtion.py file
to see explanations for all parameters.<br></br><br></br>

To create a setup I recommend using the Factories.<br></br>
Also the configuration classes make it easier to see which parameters need to be
set. <br></br>
Creater a worker configuration and use the a3c_literals defined in the 
a3c_literals.py for the named parameters.<br></br>
Give the finsihed configuration to the WorkerFactory.<br></br><br></br>


Below is an example setup which also shows how to use the shared memory.
This can also be found and executed in the example folder.<br></br>

Note that in order to use the gym environments you have to add the path of 
the gym module to the system path of your python interpreter.


<header>
<h2>
Example setup: Cartpole Environment
</h2>

</header>

```bash
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
NUM_EPISODES = 500

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
    worker_config.num_workers = 4  # mp.cpu_count()
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
```
