from multiprocessing import RawArray
from ctypes import c_double
import numpy as np
from types import ModuleType
import sys

temp_module_name = "_shared_model_"


class SharedMemory(object):

    """
    The SharedMemory class can create a piece of memory.
    The memory can be accessed (read/write) from different parallel processes.

    Example:
        If you want to access the shared memory in a running child process:
        >> w = sys.modules[temp_module_name].__dict__["w"]
        Changing the values of w changes the shared memory!
        If you don't want that to happen create a shallow copy
        >> import copy
        >> w = copy.copy(w) 
    """

    def __init__(self, size_w):

        """
        Initializes object

        Args:
            size_w(int): Total number of neural network parameters
        """

        global_theta = RawArray(c_double, np.ones((size_w, 1)))
        w = np.frombuffer(global_theta)
        w = w.reshape((len(w), 1))
        self.w = w
        self.size_w = size_w

    def __enter__(self, *args):

        # Make temporary module to store shared weights

        mod = ModuleType(temp_module_name)
        mod.__dict__['w'] = self.w
        sys.modules[mod.__name__] = mod
        self.mod = mod
        return self

    def __exit__(self, *args):

        # Clean up temporary module

        del sys.modules[self.mod.__name__]


