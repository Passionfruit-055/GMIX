import warnings

from .mylogger import init_logger
from algo.qmix import QMIXAgent
from algo.comm import CommAgent
from env import config_env

import numpy as np

logger = init_logger()


def get_config(config, key, default, warning=None):
    value = config.get(key, None)
    warning = 'get_config warning' if warning is None else warning
    if value is None:
        warnings.warn(warning)
        value = default
    return value


def set_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(config):
    env = None

    env_name = config.get('env_name', "Undefined")
    logger.info(f"ENV: {env_name}")

    map_name = config.get('map_name', None)
    logger.info(f"MAP: {map_name}")

    special_config = config_env(mode=config.get('mode', 'preset'), env=env_name, map=map_name)

    if env_name == 'mpe':
        if map_name.find('reference') != -1:
            from pettingzoo.mpe import simple_reference_v3
            # env = simple_reference_v3.env(**special_config)
            env = simple_reference_v3.parallel_env(**special_config)
            env.reset()
        else:
            raise NotImplementedError(f"Undefined MPE map <{map_name}>!")
        # update env_config of mpe
        agent_0_id = env.agents[0]
        scenario_config = {'agent_num': env.max_num_agents,
                           'obs_space': env.observation_space(agent_0_id).shape[0],
                           'action_space': env.action_space(agent_0_id).n,
                           'state_space': env.state_space.shape[0],
                           'scenario': env_name + '_' + map_name
                           }
        config.update(scenario_config)

    elif env_name == 'smac':
        # TODO load smac
        pass

    else:
        raise NotImplementedError(f"Undefined env <{env_name}>")
    return env


def match_agent(config):
    config['model'].update(config['env'])
    config['model'].update(config['experiment']['running'])
    comm_agent = CommAgent(config['model']) if config['model']['comm'] else None
    q_agent = QMIXAgent(config['model'])
    return comm_agent, q_agent


def run_one_scenario(config, seed, env, comm_agent, agent):
    results = []
    observations, infos = env.reset(seed)
    n_agent = len(env.agents)
    dones = [False for _ in range(n_agent)]
    # raw_env 是 parallel_env 的一个属性
    while env.agents:
        # this is where you would insert your policy
        done = True if True in dones else False
        observations = np.array(list(observations.values()), dtype=np.float32)

        comm_agent.communication_round(observations, done)

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
        dones = list(map(lambda x, y: x or y, terminations.values(), truncations.values()))
        results.append([observations, actions, rewards, terminations, truncations, infos, dones])
    env.close()
    return results  # return a whole episode


def store_results(episode, **results):
    pass


def plot_results(episode, **results):
    pass


def train_agent(config, comm_agent, agent):
    pass
