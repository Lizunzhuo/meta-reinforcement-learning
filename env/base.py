from gym.core import Env
import numpy as np

class MetaEnv(Env):
    """
    Wrapper around openAI gym environment,interface for meta learning
    """

    def sample_task(self,n_task):
        """
        Samples task of the meta-environment
        :param n_task (int) : number of defferent meta-tasks needed
        :return: tasks (list): an (n_task) length list of tasks
        """

        return NotImplementedError

    def set_task(self,task):
        """
        Sets the specified task to the current envuronment
        :param task: task of the meta-learning envrionment
        :return:
        """
        raise NotImplemented

    def get_task(self):
        """
        Gets the task that the agent is performing in the current environment
        :return: task: task of the meta-learning environmeng
        """
        raise NotImplemented

    def log_diagnostices(self,paths,prefix):
        """
        Logs env-specific diagnostic information
        :param paths(list): list of all paths collected with this env during this iteration
        :param prefix(str): prefix for logger
        :return:
        """
        pass