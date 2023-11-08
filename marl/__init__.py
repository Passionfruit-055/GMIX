import warnings

from .mylogger import init_logger
from algo.qmix import QMIXAgent
from algo.comm import CommAgent
from env import config_env

logger = init_logger()


def get_config(config, key, default, warning=None):
    value = config.get(key, None)
    warning = 'get_config warning' if warning is None else warning
    if value is None:
        warnings.warn(warning)
        value = default
    return value


def make_env(config):
    env = None

    env_name = config.get('env_name', "Undefined")
    logger.info(f"ENV: {env_name}")

    map_name = config.get('map_name', None)
    logger.info(f"Map: {map_name}")

    special_config = config_env(mode='preset', env=env_name, map=map_name)

    if env_name == 'mpe':
        if map_name.find('reference') != -1:
            from pettingzoo.mpe import simple_reference_v3
            env = simple_reference_v3.env(**special_config)
            seed = get_config(config, 'seed', 21, "undefined seed")
            env.reset(seed=seed)
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
    comm_agent = CommAgent(config['model']) if config['model']['comm'] else None
    q_agent = QMIXAgent(config['model'])
    return comm_agent, q_agent


def run_one_scenario(config, env, comm_agent, q_agent):
    pass
