import numpy as np
import pickle as pickle
from multiprocessing import Process,Pipe
import copy

class MetaIterativeEnvExecutor(object):
    """
    Wraps multiple environments of the same kind and provides functionlity to reset / step the environments
    in a vectorized manner. Internally, the environments are executed iteratively.

    Args:
        env (meta_policy_search.env.base.MetaEnv): meta environment object
        meta_batch_size(int): bumber of meta tasks
        envs_per_task(int): number of environments per meta task
        max_path_length(int): maximum length of sampled environment paths - if he max_path_length is reached,
                              the respective environment is reset
    """

    def __init__(self,env,meta_batch_size,envs_per_task,max_path_length):
        self.envs = np.array([copy.deepcopy(env) for _ in range(meta_batch_size * envs_per_task)])
        self.ts = np.zeros(len(self.envs),dtype='int')
        self.max_path_length = max_path_length

    def step(self,actions):
        """
        Steps the wrapped environment with the provided actions
        :param actions(list): list of action, of length meta_batch_size x envs_per_task
        :return:
                (tuple): a length 4 tuple of list, containing obs (np,array), rewards (flaot), dones (bool),
                env_infos (dict). Each list is of length meta_batch_size x envs_per_task(assumes that every task has same number of envs)
        """

        assert len(actions) == self.num_envs

        all_results = [env.step(a) for (a,env) in zip(actions, self.envs)]

        obs, rewards, dones, env_infos = list(map(list,zip(*all_results)))

        dones = np.array(dones)
        self.ts += 1
        dones = np.logical_or(self.ts >= self.max_path_length, dones)
        for i in np.argwhere(dones).flatten():
            obs[i] = self.envs[i].reset()
            self.ts[i] = 0

        return obs, rewards, dones, env_infos

    def set_tasks(self, tasks):
        """
        Set of a list of task to each environment
        :param task: list of the tasks for each environment
        :return:
        """

        envs_per_task =np.split(self.envs,len(tasks))
        for task, envs in zip(tasks,envs_per_task):
            for env in envs:
                env.set_task(task)

    def reset(self):
        """
        Resets the environment
        :return:
                (list): list of (np.ndaray) with the new initial observations.
        """
        obses = [env.reset() for env in self.envs]
        self.ts[:] = 0
        return obses

    @property
    def num_envs(self):
        """
        Number of environment
        :return:
                (int): number of environment
        """
        return len(self.envs)

class MetaParallelEnvExecutor(object):
    """
    Wraps multipe environments of the same kind and provides functionality to reset / step the environments
    in a vectorized manner. Thereby the environment are distribute among meta_batch_size processes and executed in parallel.

    Args:
        env(meta_policy_search.envs.base,MetaEnv): meta environment object
        meta_batch_size(int): number of environments per meta task
        envs_per_task(int): number of environments per meta task
        mas_path_length(int): maximum length of sampled environment paths - if the max_path_length is reached, the respective environment is reset
    """

    def __init__(self,env, meta_batch_size, envs_per_task, max_path_length):
        self.n_envs =meta_batch_size * envs_per_task
        self.meta_batch_size = meta_batch_size
        self.envs_per_task = envs_per_task
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(meta_batch_size)])
        seeds = np.random.choice(range(10**6), size=meta_batch_size, replace=False)

        self.ps = [
            Process(target=worker, args=(work_remote, pickle.dumps(env), envs_per_task,max_path_length, seed))
            for (work_remote, remote, seed) in zip(self.work_remotes, self.remotes, seeds)
        ]


        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()


def worker(remote, parent_remote, env_pickle, n_envs, max_path_length, seed):
    """
    
    :param remote:
    :param parent_remote:
    :param env_pickle:
    :param n_envs:
    :param max_path_length:
    :param seed:
    :return:
    """