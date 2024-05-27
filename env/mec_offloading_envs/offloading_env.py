from env.base import MetaEnv
from env.mec_offloading_envs.offloading_task_graph import OffloadingTaskGraph
from samplers.vectorized_env_executor import MetaIterativeEnvExecutor

import numpy as np
import os

class Resources(object):
    """
    This class denotes the MEC server and Mobile devices (computation resources)

    Args:
        mec_process_capable: computation capacity of the MEC server
        mobile_process_capable: computation capacity of the mobile device
        bandwidth_up: wireless uplink band width
        bandwidth_dl: wireless downlink band width
    """

    def __init__(self, mec_process_capable, mobile_process_capable, bandwidth_up, bandwidth_dl):
        self.mec_process_capable = mec_process_capable
        self.mobile_process_capable = mobile_process_capable
        self.mobile_process_avaliable_time = 0.0
        self.mec_process_avaliable_time = 0.0

        self.bandwidth_up = bandwidth_up
        self.bandwidth_dl = bandwidth_dl

    def up_transmission_cost(self, data):
        rate = self.bandwidth_up * (1024.0 * 1024.0 / 8.0)
        transmission_time = data / rate

        return transmission_time

    def reset(self):
        self.mobile_process_avaliable_time = 0.0
        self.mec_process_avaliable_time = 0.0

    def dl_transmission_cost(self, data):
        rate = self.bandwidth_dl * (1024.0 *1024.0 /8.0)
        transmission_time = data / rate

        return transmission_time

    def locally_execution_cost(self, data):
        return self._computation_cost(data, self.mobile_process_capable)

    def mec_execution_cost(self, data):
        return self._computation_cost(data, self.mec_process_capable)

    def _computation_cost(self, data, processing_power):
        computation_time = data / processing_power

        return computation_time