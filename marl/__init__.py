import warnings
import numpy as np
from datetime import datetime
import random
import logging
import os
import scipy.io as sio
import matplotlib.pyplot as plt

from .mylogger import init_logger
from algo.qmix import QMIXAgent
from algo.comm import CommAgent
from env import config_env

now = datetime.now()
today = now.strftime("%m.%d")
current_time = now.strftime("%H_%M")
rootpath = './results/' + today + '/'
folder = ['/' + f + '/' for f in ['data', 'fig', 'video']]


def count_folders(path):
    folder_count = 0
    # for _, dirs, _ in os.walk(path[:-1]):
    #     folder_count += len(dirs)
    for fn in os.listdir(path[:-1]):  # fn 表示的是文件名
        folder_count += 1

    return folder_count


def basic_preparation(config, info):
    def _name_batch():
        env = config['env']['env_name']
        mapn = config['env']['map_name']
        batch_count = count_folders(rootpath)
        return '/' + str(batch_count) + '_' + env.upper() + mapn + '__' + info + '/'

    batch_name = _name_batch()
    # prepare logger
    logger = init_logger(config['experiment']['logger'], log_path=rootpath + batch_name)
    logger.info(f"Run it for: {info}")
    # set seed
    set_seed(config['experiment']['running'].get('seed', 21))
    seed = random.randint(0, 1000)
    logger.debug(f"seed: {seed}")

    episode = config['experiment']['running'].get('episode', 1000)
    seq_len = config['experiment']['running'].get('timestep', 100)

    # create result folder
    for f in folder:
        if not os.path.exists(rootpath + batch_name + f):
            os.makedirs(rootpath + batch_name + f)
    return logger, batch_name, seed, episode, seq_len


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
    logger = logging.getLogger()

    env_name = config.get('env_name', "Undefined")
    logger.info(f"ENV: {env_name}")

    map_name = config.get('map_name', None)
    logger.info(f"MAP: {map_name}")

    env_config = config_env(mode=config.get('mode', 'preset'), env=env_name, map=map_name)

    if env_name == 'mpe':
        if map_name.find('reference') != -1:
            from pettingzoo.mpe import simple_reference_v3
            # env = simple_reference_v3.env(**special_config)
            env = simple_reference_v3.parallel_env(**env_config)
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
    agent.reset_hidden_state()
    # raw_env 是 parallel_env 的一个属性
    while env.agents:
        # this is where you would insert your policy
        done = True if True in dones else False
        observations = np.array(list(observations.values()), dtype=np.float32)

        e_obs = comm_agent.communication_round(observations, done)

        actions, warning_signals = agent.choose_actions(e_obs)

        actions = {agent: action for agent, action in zip(env.agents, actions)}

        observations, rewards, terminations, truncations, infos = env.step(actions)
        dones = list(map(lambda x, y: x or y, terminations.values(), truncations.values()))

        results.append([e_obs, actions, rewards, warning_signals, dones])

    env.close()
    return results  # return a whole episode


def store_results(episode, batch_name, agent, results):
    n_agent = agent.n_agent
    seq_len = len(results)

    obs, actions, rewards, mus, dones = [], [], [], [], []
    for result in results:
        for elem, category in zip(result, [obs, actions, rewards, mus, dones]):
            if isinstance(elem, dict):
                values = np.array(list(elem.values()))
            elif isinstance(elem, list):
                values = np.array(elem)
            elif isinstance(elem, np.ndarray):
                values = elem
            category.append(values)

    # to ndarray (T, N, size)
    obs = np.array(obs).reshape(seq_len, n_agent, -1)
    actions = np.array(actions).reshape(seq_len, n_agent, -1)
    rewards = np.array(rewards).reshape(seq_len, n_agent, -1)
    mus = np.array(mus).reshape(seq_len, n_agent, -1)
    dones = np.array(dones).reshape(seq_len, n_agent, -1)

    truncations = []
    for d in np.array(dones, dtype=np.int32).reshape(n_agent, seq_len):
        bp = np.where(d == 1)
        truncations.append(bp[0][0]) if len(bp[0]) > 0 else truncations.append(seq_len)

    def _formulate_system_state(obs, seq_len):
        return obs.copy().reshape(seq_len, -1)

    states = _formulate_system_state(obs, seq_len)
    # n_obs和n_states要在结尾填充一条记录
    n_obs = np.append(obs[1:].copy(), obs[-1]).reshape(seq_len, n_agent, -1)
    n_states = np.append(states[1:].copy(), states[-1]).reshape(seq_len, -1)

    agent.store_experience(obs, actions, rewards, n_obs, dones, states, n_states, mus)

    def save_to_mat():
        filenames = ['rewards.mat', 'actions.mat', 'mus.mat', 'dones.mat']
        for filename, category in zip(filenames, [rewards, actions, mus, dones]):
            if not os.path.exists(rootpath + batch_name + folder[0] + filename):
                sio.savemat(rootpath + batch_name + folder[0] + filename, {f'episode{episode}': category})
            else:
                history = sio.loadmat(rootpath + batch_name + folder[0] + filename)
                history.update({f'episode{episode}': category})
                sio.savemat(rootpath + batch_name + folder[0] + filename, history)

    save_to_mat()

    return obs, actions, rewards, n_obs, dones, states, n_states, mus


def plot_results(episode, results, config):
    # process data
    obs, actions, rewards, n_obs, dones, states, n_states, mus = results
    n_agent = rewards.shape[1]

    def _global_reward():
        all_rewards = np.sum(rewards.squeeze(), axis=1)
        return all_rewards / len(n_agent)

    rewards = rewards.reshape(n_agent, -1)
    global_reward = _global_reward()

    def _set_canvas():
        plt.style.use(config.get('theme', 'seaborn'))
        plt.rcParams['font.family'] = config.get('font_family', 'Times New Roman')
        plt.rcParams['font.size'] = config.get('fontsize', 15)
        cmap = plt.cm.get_cmap(config.get('cmap', 'Set2'))
        color = cmap.colors
        return color

    colors = _set_canvas()

    def _rewards():
        fig, axes = plt.subplots(1, n_agent + 1, figsize=((n_agent + 1) * 6 + 2, 6))
        xlabels = [f'Agent{i + 1}' for i in range(n_agent)] + ['Global']
        for ax, xlabel, reward, color in zip(axes, xlabels, (rewards[0], rewards[1], global_reward), colors):
            ax.plot(reward, color=color, label=xlabel)
            ax.set_xlabel('Timestep')
        plt.legend()
        plt.title('Reward')
        plt.tight_layout()
        plt.savefig(rootpath + folder[1] + 'rewards.png')
        plt.savefig(rootpath + folder[1] + 'rewards.pdf')
        plt.close()

    def _reward_in_one():
        plt.figure()
        labels = [f'Agent{i + 1}' for i in range(n_agent)] + ['Global']
        for label, reward, color in zip(labels, (rewards[0], rewards[1], global_reward), colors):
            plt.plot(reward, color=color, label=label)
        plt.legend()
        plt.tight_layout()
        plt.savefig(rootpath + folder[1] + 'rewards_in_one.png')
        plt.savefig(rootpath + folder[1] + 'rewards_in_one.pdf')
        plt.close()

    _rewards()
    _reward_in_one()


def train_agent(config, comm_agent, agent):
    pass
