import warnings
import numpy as np
from datetime import datetime
import random
import logging
import os
import scipy.io as sio
import torch

from .mylogger import init_logger
from algo.qmix import QMIXAgent, losses
from algo.comm import CommAgent
from env import config_env
import matplotlib.pyplot as plt

# from marl.plot import plot_results

now = datetime.now()
today = now.strftime("%m.%d")
current_time = now.strftime("%H_%M")
rootpath = './results/' + today + '/'
if not os.path.exists(rootpath):
    os.makedirs(rootpath)
folder = ['/' + f + '/' for f in ['data', 'fig', 'video']]

save_this_batch = True

tot_episode = 0
seq_len = 0

batch_supervisor = iter([0])
exp_name = ''

logger = None

end_rewards = []

env_name = ''
map_name = ''

need_guide, communicate = True, True


def _count_folders(path):
    folder_count = 0
    for _ in os.listdir(path[:-1]):
        folder_count += 1
    return folder_count


def running_config(config, info):
    batches = config.get('experiment', {}).get('running', {}).get('batches', [0])
    batch_num = len(batches)

    running_mode = config['experiment']['running']['mode']

    def _batch_generator(batches):
        for batch in batches:
            yield batch

    global batch_supervisor
    batch_supervisor = _batch_generator(batches)

    def _name_batch():
        env = config['env']['env_name']
        Map = config['env']['map_name']
        batch_count = _count_folders(rootpath)
        return '/' + str(batch_count) + '_' + env.upper() + Map + '_' + info + '/'

    global exp_name
    exp_name = _name_batch()

    global save_this_batch
    save_this_batch = False if info.find('test') != -1 or running_mode == 'debug' else True

    # prepare logger
    global logger
    logger = init_logger(config['experiment']['logger'],
                         log_path=rootpath + exp_name if save_this_batch else rootpath)
    logger.info(f"\nRun it for: {info}")
    logger.info(f"Current experiment is{'' if save_this_batch else ' NOT'} recorded!")

    # set seed
    seed = set_seed(config['experiment']['running'].get('seed', 21))
    logger.debug(f"seed: {seed}")

    if save_this_batch:
        # create result folder for this experiment
        if not os.path.exists(rootpath + exp_name):
            os.makedirs(rootpath + exp_name)

    global tot_episode, seq_len
    tot_episode = config['experiment']['running'].get('episode', 1000)
    seq_len = config['experiment']['running'].get('timestep', 100)

    global end_rewards
    end_rewards = {str(batch): {} for batch in batches}

    return batch_num, seed, tot_episode, seq_len, logger


def one_batch_basic_preparation(config, env):
    env.reset()

    this_batch = next(batch_supervisor)
    logger.info(f"This batch is for {this_batch}")

    def _parse_this_batch():
        if this_batch.find('mix') != -1:
            config['model'].update({'guide': False, 'comm': False})
            if this_batch.find('g') != -1:
                config['model'].update({'guide': True})
                logger.info(f"add guide")
            if this_batch.find('c') != -1:
                config['model'].update({'comm': True})
                logger.info(f"add communication")

    _parse_this_batch()

    global need_guide, communicate
    need_guide = config['model'].get('guide', True)
    communicate = config['model'].get('comm', True)

    batch_name = exp_name + '/' + str(this_batch) + '/'

    if save_this_batch:
        # create result folder for this batch
        for f in folder:
            if not os.path.exists(rootpath + batch_name + f):
                os.makedirs(rootpath + batch_name + f)

    return batch_name


def get_config(config, key, default, warning=None):
    value = config.get(key, None)
    warning = 'get_config warning' if warning is None else warning
    if value is None:
        warnings.warn(warning)
        value = default
    return value


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def make_env(config):
    env = None

    env_name = config.get('env_name', "Undefined")
    logger.info(f"ENV: {env_name}")

    map_name = config.get('map_name', None)
    logger.info(f"MAP: {map_name}\n")

    env_config = config_env(mode=config.get('param', 'preset'), env=env_name, map=map_name)

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
    algo = config.get('model', {}).get('algo', 'qmix')
    comm_agent = CommAgent(config['model']) if communicate else None
    if algo == 'qmix':
        q_agent = QMIXAgent(config['model'])
    else:
        raise NotImplementedError(f"Algorithm {algo} undefined!")
    return comm_agent, q_agent


def run_one_scenario(config, seed, env, comm_agent, agent):
    results = []
    observations, infos = env.reset(seed)
    n_agent = len(env.agents)
    dones = [False for _ in range(n_agent)]
    agent.reset_hidden_state(seq_len=1)
    # raw_env 是 parallel_env 的一个属性
    while env.agents:
        done = False if False in dones else True
        observations = np.array(list(observations.values()), dtype=np.float32)

        e_obs = comm_agent.communication_round(observations, done) if communicate else torch.from_numpy(
            observations).to(agent.device)

        actions, warning_signals = agent.choose_actions(e_obs)

        observations, rewards, terminations, truncations, infos = env.step(
            {agent: action for agent, action in zip(env.agents, actions)})  # 传进去的需要是一个字典

        dones = list(map(lambda x, y: x or y, terminations.values(), truncations.values()))

        results.append([e_obs, actions, rewards, warning_signals, dones])

    env.close()

    return results  # return a whole episode


def store_results(episode, batch_name, agent, results):
    n_agent = agent.n_agent
    seq_len = len(results)

    obs, actions, rewards, mus, dones = [], [], [], [], []
    for result in results:
        for elem, category, c_name in zip(result, [obs, actions, rewards, mus, dones],
                                          ['obs', 'actions', 'rewards', 'mus', 'dones']):
            if isinstance(elem, dict):
                values = np.array(list(elem.values()))
            elif isinstance(elem, list):
                values = np.array(elem)
            elif isinstance(elem, np.ndarray):
                values = elem
            elif isinstance(elem, torch.Tensor):
                values = elem.detach().cpu().numpy()
            elif elem is None:
                values = None
            else:
                raise TypeError(f"{c_name}'s type {type(elem)} need process!")
            category.append(values)

    # to ndarray (T, N, size)
    obs = np.array(obs).reshape(seq_len, n_agent, -1)
    actions = np.array(actions, dtype=np.int64).reshape(seq_len, n_agent, -1)
    rewards = np.array(rewards).reshape(seq_len, n_agent, -1)
    mus = np.array(mus).reshape(seq_len, n_agent, -1) if need_guide else np.zeros((seq_len, n_agent, 1))
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

    def _save_to_mat():
        filenames = ['rewards.mat', 'actions.mat', 'mus.mat', 'dones.mat']
        for filename, category in zip(filenames, [rewards, actions, mus, dones]):
            if not os.path.exists(rootpath + batch_name + folder[0] + filename):
                sio.savemat(rootpath + batch_name + folder[0] + filename, {f'episode{episode}': category})
            else:
                history = sio.loadmat(rootpath + batch_name + folder[0] + filename)
                history.update({f'episode{episode}': category})
                sio.savemat(rootpath + batch_name + folder[0] + filename, history)

    if save_this_batch:
        _save_to_mat()

    return obs, actions, rewards, n_obs, dones, states, n_states, mus


def train_agent(config, comm_agent, agent):
    # comm_agent 每一个communication round都会进行train, 这里仅考虑GMIX的train过程
    autograd_detect = not save_this_batch
    agent.train(autograd_detect)


def plot_results(episode, batch_name, results, config):
    obs, actions, rewards, n_obs, dones, states, n_states, mus = results

    algo = batch_name.split('/')[-2]

    end_reward = end_rewards[algo]

    n_agent = rewards.shape[1]
    labels = [f'Agent{i + 1}' for i in range(n_agent)] + ['Global']

    def _save_mode():
        keep_live = config.get('keep_live', True)
        if keep_live:
            live_path = rootpath + batch_name + folder[1] + '/live/'
            if not os.path.exists(live_path):
                os.mkdir(live_path)
        save_recent_n_episode = config.get('save_recent_n_episode', 32)
        return keep_live, save_recent_n_episode

    if save_this_batch:
        live, recent_num = _save_mode()

    def _global_reward():
        all_rewards = np.sum(rewards.squeeze(), axis=0)
        return all_rewards / n_agent

    rewards = rewards.reshape(n_agent, -1)
    global_reward = _global_reward()

    def _extract_end_reward():
        for i, r in enumerate(rewards):
            if len(end_reward) < i + 1:
                end_reward.update({f'Agent{i}': [r[-1]]})
            else:
                end_reward[f'Agent{i}'].append(r[-1])
        if len(end_reward) < n_agent + 1:
            end_reward.update({'Global': [global_reward[-1]]})
        else:
            end_reward['Global'].append(global_reward[-1])

    _extract_end_reward()

    def _set_canvas():
        plt.style.use(config.get('theme', 'seaborn'))
        plt.rcParams['font.family'] = config.get('font_family', 'Times New Roman')
        plt.rcParams['font.size'] = config.get('fontsize', 15)
        cmap = plt.colormaps.get_cmap(config.get('cmap', 'Set2'))
        color = cmap.colors
        return color

    colors = _set_canvas()

    def _rewards():
        fig, axes = plt.subplots(1, n_agent + 1, figsize=((n_agent + 1) * 6 + 2, 6))
        labels = [f'Agent{i + 1}' for i in range(n_agent)] + ['Global']
        for ax, label, reward, color in zip(axes, labels, (rewards[0], rewards[1], global_reward), colors):
            ax.plot(reward, color=color, label=label)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Reward')
            ax.legend()
        fig.suptitle('Episode ' + str(episode))
        plt.tight_layout()
        if save_this_batch:
            plt.savefig(rootpath + batch_name + folder[1] + f'/live/{episode % recent_num}_rewards.png')
            plt.savefig(rootpath + batch_name + folder[1] + f'/live/{episode % recent_num}_rewards.pdf')
        else:
            pass
            # plt.show()
        plt.close()

    def _reward_in_one():
        plt.figure()
        for label, reward, color in zip(labels, (rewards[0], rewards[1], global_reward), colors):
            plt.plot(reward, color=color, label=label)
        plt.legend()
        plt.xlabel('Timestep')
        plt.ylabel('Reward')
        plt.title('Episode ' + str(episode))
        plt.tight_layout()
        if save_this_batch:
            plt.savefig(rootpath + batch_name + folder[1] + f'/live/{episode % recent_num}_rewards_in_one.png')
            plt.savefig(rootpath + batch_name + folder[1] + f'/live/{episode % recent_num}_rewards_in_one.pdf')
        else:
            pass
            # plt.show()
        plt.close()

    def _losses():
        plt.figure()
        plt.xlabel('Timestep')
        plt.ylabel('Losses')
        plt.title('Until episode ' + str(episode))
        plt.plot(losses, color=random.choice(colors))
        if save_this_batch:
            plt.savefig(rootpath + batch_name + folder[1] + f'/losses.png')
            # plt.savefig(rootpath + batch_name + folder[1] + f'/losses.pdf')
        else:
            pass
            # if episode % (tot_episode // 10) == 0:
            #     plt.show()
        plt.close()

    def _end_reward():
        plt.figure()
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        for end_r, label, color in zip(end_reward.values(), labels, colors):
            plt.plot(end_r, label=label, color=color)
        plt.legend()
        plt.tight_layout()
        if save_this_batch:
            plt.savefig(rootpath + batch_name + folder[1] + f'/EpisodeReward.png')
            # plt.savefig(rootpath + batch_name + folder[1] + f'/EpisodeReward.pdf')
        else:
            pass
            # if episode % (tot_episode // 10) == 0:
            #     plt.show()
        plt.close()

    _rewards()
    _reward_in_one()
    _end_reward()
    _losses()


def exp_summary():
    summary_path = rootpath + exp_name + '/summary/'
    if not os.path.exists(summary_path):
        os.mkdir(summary_path)

    last_rewards = {}

    plt.figure()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    colors = plt.colormaps.get_cmap('Set3').colors
    for algo, color in zip(end_rewards.keys(), colors):
        last_rewards.update({algo: end_rewards[algo]['Global']})
        plt.plot(end_rewards[algo]['Global'], label=algo, color=random.choice(colors))
    plt.legend()
    plt.tight_layout()
    if save_this_batch:
        plt.savefig(summary_path + f'/Reward.png')
        plt.savefig(summary_path + f'/Reward.pdf')

        # store the mete data of this pic
        sio.savemat(summary_path + '/rewards.mat', last_rewards)
    else:
        pass
        # if episode % (tot_episode // 10) == 0:
        #     plt.show()
    plt.close()


