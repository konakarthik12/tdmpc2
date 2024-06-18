from gym.wrappers import TimeLimit, StepAPICompatibility
from .env import OmnigibEnv

from .tasks import tasks

def make_env(cfg):
    domain, task_name = cfg.task.replace('-', '_').split('_', 1)
    # print("Domain: ", domain)
    # print("Task: ", task)
    if domain != "fetch":
        raise ValueError('Unknown domain:', domain)

    if task_name not in tasks:
        raise ValueError('OmniGibson Fetch Robot - Unknown task:', task_name)

    if cfg.get('obs', 'state') != 'rgb':
        raise ValueError('OmniGibson Fetch Robot only supports rgb observation, not:', cfg.get('obs', 'state'))

    # omni_env_cfg = task.task_config(cfg)

    # omni_env = OmniGibsonEnv(omni_env_cfg)
    omni_env = tasks[task_name].make_env(cfg)
    max_env_steps = cfg.get('max_env_steps', 1000)
    omni_env = TimeLimit(omni_env, max_episode_steps=max_env_steps)
    omni_env = StepAPICompatibility(omni_env, output_truncation_bool=False)
    return omni_env
