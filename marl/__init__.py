import logging
import warnings
import numpy as np
from datetime import datetime
import random
import os
import scipy.io as sio
import torch
from collections import deque

from .mylogger import init_logger
from algo.qmix import QMIXAgent
from algo.comm import CommAgent
from env import config_env
import matplotlib.pyplot as plt
from .cache import Cache

now = datetime.now()
today = now.strftime("%m.%d")
current_time = now.strftime("%H_%M")
rootpath = './results/' + today + '/'
if not os.path.exists(rootpath):
    os.makedirs(rootpath)
folder = ['/' + f + '/' for f in ['data', 'fig', 'replay', 'tensorboard']]

save_this_batch = True
save_pdf = True

tensorboard_writer = None

tot_episode = 0
seq_len = 0

batch_supervisor = iter([0])
exp_name = ''

logger = logging.getLogger()

env_name = ''
map_name = ''

need_guide, communicate = True, True

batch_name = ''
this_batch_algo = ''

exp_cache = Cache('store results of different batches')
exp_cache['rewards'] = {}
exp_cache['dangers'] = {}
exp_cache['comm_rewards'] = {}

batch_cache = Cache('store results of different episodes')
batch_cache['comm_rewards'] = deque(maxlen=int(1e4))
batch_cache['loss'] = deque(maxlen=int(1e4))


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

    global save_pdf
    save_pdf = config['experiment']['running'].get('save_pdf', False)

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

    global exp_cache
    exp_cache['rewards'] = {str(batch): {} for batch in batches}

    return batch_num, seed, tot_episode, seq_len, logger


def one_batch_basic_preparation(config, env):
    env.reset()

    global this_batch_algo
    this_batch = str(next(batch_supervisor))
    this_batch_algo = this_batch
    logger.info(f"This batch is for {this_batch}")

    def _cache():
        global batch_cache
        batch_cache['comm_rewards'].clear()
        batch_cache['loss'].clear()

        global exp_cache
        exp_cache['dangers'][this_batch] = {}

    _cache()

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

    global batch_name
    batch_name = exp_name + '/' + str(this_batch) + '/'

    def _launch_tensorboard():
        from torch.utils.tensorboard import SummaryWriter
        global tensorboard_writer
        tensorboard_writer = SummaryWriter(rootpath + batch_name + folder[3])
        os.system(f"tensorboard --logdir={rootpath + batch_name + folder[3]}")

    if save_this_batch:
        # create result folder for this batch
        for f in folder:
            if not os.path.exists(rootpath + batch_name + f):
                os.makedirs(rootpath + batch_name + f)

        # _launch_tensorboard()


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
            # from pettingzoo.mpe import simple_reference_v3
            from env.mpe_scenario import reference_risk
            # env = simple_reference_v3.env(**special_config)
            env = reference_risk.parallel_env(**env_config)
            env.reset()
        else:
            raise NotImplementedError(f"Undefined MPE map <{map_name}>!")
        # update env_config of mpe
        agent_0_id = env.agents[0]
        scenario_config = {'agent_num': env.max_num_agents,
                           'obs_space': env.observation_space(agent_0_id).shape[0],
                           'action_space': env.action_space(agent_0_id).n,
                           'state_space': env.state_space.shape[0],
                           'mpe_scenario': env_name + '_' + map_name
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


def run_one_episode(seed, env, comm_agent, agent):
    results = []
    observations, infos = env.reset(seed)
    n_agent = len(env.agents)
    dones = [False for _ in range(n_agent)]
    agent.reset_hidden_state(seq_len=1)
    rewards = np.zeros((n_agent, 1))

    while env.agents:
        done = False if False in dones else True
        observations = np.array(list(observations.values()), dtype=np.float32)

        obs_new, comm_reward = comm_agent.communication_round(observations, rewards.copy(), done) if communicate else (
            torch.from_numpy(observations).to(agent.device), None)
        global batch_cache
        if comm_reward is not None:
            for c_r in comm_reward:
                batch_cache['comm_rewards'].append(c_r)

        actions, warning_signals = agent.choose_actions(obs_new)

        observations, rewards, terminations, truncations, infos = env.step(
            {agent: action for agent, action in zip(env.agents, actions)})  # 传进去的需要是一个字典
        danger_times = env.unwrapped.world.danger_infos()

        utilities = agent.compute_utility(rewards, warning_signals)
        dones = list(map(lambda x, y: x or y, terminations.values(), truncations.values()))

        results.append([obs_new, actions, utilities, warning_signals, dones, danger_times])

    env.close()

    return results  # return a whole episode


def store_results(episode, agent, results):
    n_agent = agent.n_agent
    seq_len = len(results)
    obs, actions, rewards, mus, dones, danger_times, agents_in_danger = [], [], [], [], [], [], []
    for result in results:
        for elem, category, c_name in zip(result, [obs, actions, rewards, mus, dones, danger_times],
                                          ['obs', 'actions', 'rewards', 'mus', 'dones', 'danger_times']):
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
    mus = np.array(mus).reshape(seq_len, n_agent, -1)
    dones = np.array(dones).reshape(seq_len, n_agent, -1)
    danger_times = np.array(danger_times).reshape(seq_len, -1)

    truncations = []
    for d in np.array(dones, dtype=np.int32).reshape(n_agent, seq_len):
        bp = np.where(d == 1)
        truncations.append(bp[0][0]) if len(bp[0]) > 0 else truncations.append(seq_len)

    def _formulate_system_state(obs, seq_len):
        return obs.copy().reshape(seq_len, -1)

    states = _formulate_system_state(obs, seq_len)
    # n_obs 和 n_states 要在结尾填充一条记录
    n_obs = np.append(obs[1:].copy(), obs[-1]).reshape(seq_len, n_agent, -1)
    n_states = np.append(states[1:].copy(), states[-1]).reshape(seq_len, -1)

    agent.store_experience(obs, actions, rewards, n_obs, dones, states, n_states, mus)

    def _save_to_mat():
        filenames = ['rewards.mat', 'actions.mat', 'mus.mat', 'dones.mat', 'danger_times.mat']
        for filename, category in zip(filenames, [rewards, actions, mus, dones, danger_times]):
            if not os.path.exists(rootpath + batch_name + folder[0] + filename):
                sio.savemat(rootpath + batch_name + folder[0] + filename, {f'episode{episode}': category})
            else:
                history = sio.loadmat(rootpath + batch_name + folder[0] + filename)
                history.update({f'episode{episode}': category})
                sio.savemat(rootpath + batch_name + folder[0] + filename, history)

        keys = ['rewards', 'actions', 'mus', 'dones', 'danger_times']
        if not os.path.exists(rootpath + batch_name + folder[0] + '/results.mat'):
            episode_result = {key: [value] for key, value in zip(keys, [rewards, actions, mus, dones, danger_times])}
            sio.savemat(rootpath + batch_name + folder[0] + '/results.mat', episode_result)
        else:
            history = sio.loadmat(rootpath + batch_name + folder[0] + '/results.mat')
            for key, value in zip(keys, [rewards, actions, mus, dones, danger_times]):
                history[key].append(value)
            sio.savemat(rootpath + batch_name + folder[0] + '/results.mat', history)

    if save_this_batch:
        _save_to_mat()

    def _store_cache():
        global exp_cache, batch_cache
        if this_batch_algo.find('c') != -1:
            global exp_cache
            exp_cache['comm_rewards'].update({this_batch_algo: batch_cache['comm_rewards']})

        def _global_reward():
            all_rewards = np.sum(rewards.squeeze(), axis=0)
            return all_rewards / n_agent

        nonlocal rewards
        rewards = rewards.reshape(n_agent, -1)
        global_reward = _global_reward()

        end_reward = exp_cache['rewards'][this_batch_algo]

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

    _store_cache()
    #
    return obs, actions, rewards, n_obs, dones, states, n_states, mus, danger_times


def train_agent(comm_agent, agent):
    autograd_detect = not save_this_batch
    loss = agent.train(autograd_detect)
    batch_cache['loss'].extend(loss)


def plot_results(episode, results, config):
    obs, actions, rewards, n_obs, dones, states, n_states, mus = results

    algo = this_batch_algo

    n_agent = rewards.shape[0]
    labels = [f'Agent{i}' for i in range(n_agent)] + ['Global']

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

    def _set_canvas():
        plt.style.use(config.get('theme', 'seaborn'))
        plt.rcParams['font.family'] = config.get('font_family', 'Times New Roman')
        plt.rcParams['font.size'] = config.get('fontsize', 15)
        cmap = plt.colormaps.get_cmap(config.get('cmap', 'Set2'))
        color = cmap.colors
        return color

    colors = _set_canvas()

    def _global_reward():
        all_rewards = np.sum(rewards.squeeze(), axis=0)
        return all_rewards / n_agent

    rewards = rewards.reshape(n_agent, -1)
    global_reward = _global_reward()

    def _rewards():
        fig, axes = plt.subplots(1, n_agent + 1, figsize=((n_agent + 1) * 6 + 2, 6))
        for ax, label, reward, color in zip(axes, labels, (rewards[0], rewards[1], global_reward), colors):
            ax.plot(reward, color=color, label=label)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Reward')
            ax.legend()
        fig.suptitle(algo.upper() + ' Episode ' + str(episode))
        plt.tight_layout()
        if save_this_batch:
            plt.savefig(rootpath + batch_name + folder[1] + f'/live/rewards_{episode % recent_num}.png')
            if save_pdf:
                plt.savefig(rootpath + batch_name + folder[1] + f'/live/rewards_{episode % recent_num}.pdf')
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
        plt.title(algo.upper() + ' Episode ' + str(episode))
        plt.tight_layout()
        if save_this_batch:
            plt.savefig(rootpath + batch_name + folder[1] + f'/live/rewards_{episode % recent_num}_in_one.png')
            if save_pdf:
                plt.savefig(rootpath + batch_name + folder[1] + f'/live/rewards_{episode % recent_num}_in_one.pdf')
        else:
            pass
            # plt.show()
        plt.close()

    def _losses():
        plt.figure()
        plt.xlabel('Timestep')
        plt.ylabel('Loss')
        plt.title(algo.upper() + ' Until episode ' + str(episode))
        plt.plot(batch_cache['loss'], color=random.choice(colors))
        if save_this_batch:
            plt.savefig(rootpath + batch_name + folder[1] + f'/loss.png')
            if save_pdf:
                plt.savefig(rootpath + batch_name + folder[1] + f'/loss.pdf')
        else:
            pass
            # if episode % (tot_episode // 10) == 0:
            #     plt.show()
        plt.close()

    def _end_reward():
        plt.figure()
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        for end_r, label, color in zip(exp_cache['rewards'][algo].values(), labels, colors):
            plt.plot(end_r, label=label, color=color)
        plt.legend()
        plt.tight_layout()
        plt.title(algo.upper())
        if save_this_batch:
            plt.savefig(rootpath + batch_name + folder[1] + f'/EpisodeReward.png')
            if save_pdf:
                plt.savefig(rootpath + batch_name + folder[1] + f'/EpisodeReward.pdf')
        else:
            pass
            # if episode % (tot_episode // 10) == 0:
            #     plt.show()
        plt.close()

    def _dangers():
        mus_all = np.sum(mus, axis=1)
        fig, axes = plt.subplots(1, n_agent + 1, figsize=((n_agent + 1) * 6 + 2, 6))
        for ax, label, danger, color in zip(axes, labels, [mus[0], mus[1], mus_all], colors):
            ax.plot(danger, color=color, label=label)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Dangers')
            ax.legend()
        fig.suptitle(algo + ' Episode ' + str(episode))
        plt.tight_layout()
        if save_this_batch:
            plt.savefig(rootpath + batch_name + folder[1] + f'/live/dangers_{episode % recent_num}.png')
            if save_pdf:
                plt.savefig(rootpath + batch_name + folder[1] + f'/live/dangers_{episode % recent_num}.pdf')
        else:
            pass
            # plt.show()
        plt.close()

        global exp_cache
        for label, danger in zip(labels, [mus[:, 0, :].sum(), mus[:, 1, :].sum(), mus_all.sum()]):
            if label not in exp_cache['dangers'][algo].keys():
                exp_cache['dangers'][algo][label] = deque(maxlen=int(1e4))
            exp_cache['dangers'][algo][label].append(danger)

    def _end_dangers():
        # times reaches the danger zone per episode
        plt.figure()
        plt.xlabel('Episode')
        plt.ylabel('Times reaches danger zones')
        for danger, label, color in zip(exp_cache['dangers'][algo].values(), labels, colors):
            plt.plot(danger, label=label, color=color)
        plt.legend()
        plt.title(algo.upper())
        plt.tight_layout()
        if save_this_batch:
            plt.savefig(rootpath + batch_name + folder[1] + f'/EpisodeDanger.png')
            if save_pdf:
                plt.savefig(rootpath + batch_name + folder[1] + f'/EpisodeDanger.pdf')
        else:
            pass
            # if episode % (tot_episode // 10) == 0:
            #     plt.show()
        plt.close()

    def _comm_reward():
        # 因为没有一个 end 的情况，整体的 batch 可看做是一个连续的训练过程，所以把整体的 comm reward 画在一张图里
        plt.figure()
        plt.xlabel('Episode')
        plt.ylabel('Communication Reward')
        plt.plot(batch_cache['comm_rewards'], color=random.choice(colors))
        plt.title(algo.upper())
        plt.tight_layout()
        if save_this_batch:
            plt.savefig(rootpath + batch_name + folder[1] + f'/CommReward.png')
            if save_pdf:
                plt.savefig(rootpath + batch_name + folder[1] + f'/CommReward.pdf')
        else:
            pass
            # if episode % (tot_episode // 10) == 0:
            #     plt.show()
        plt.close()

    _rewards()
    _reward_in_one()
    _end_reward()

    _losses()

    _dangers()
    _end_dangers()

    if algo.find('c') != -1:
        _comm_reward()


def exp_summary():
    def _reward_summary():
        last_rewards = {}
        plt.figure()
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        colors = plt.colormaps.get_cmap('tab10').colors
        for algo, color in zip(exp_cache['rewards'].keys(), colors):
            last_rewards.update({algo: exp_cache['rewards'][algo]['Global']})
            plt.plot(exp_cache['rewards'][algo]['Global'], label=algo, color=color, linewidth=1.5)
        plt.legend()
        plt.tight_layout()
        if save_this_batch:
            summary_path = rootpath + exp_name + '/summary/'
            if not os.path.exists(summary_path):
                os.mkdir(summary_path)
            plt.savefig(summary_path + f'/Reward.png')
            if save_pdf:
                plt.savefig(summary_path + f'/Reward.pdf')
            sio.savemat(summary_path + '/rewards.mat', last_rewards)
        else:
            plt.show()
        plt.close()

    def _danger_summary():
        all_dangers = {}
        colors = plt.colormaps.get_cmap('Set1').colors
        plt.figure()
        plt.xlabel('Episode')
        plt.ylabel('Times reach danger zones')
        for algo, color in zip(exp_cache['dangers'].keys(), colors):
            all_dangers.update({algo: exp_cache['dangers'][algo]['Global']})
            plt.plot(exp_cache['dangers'][algo]['Global'], label=algo, color=color, linewidth=1.5)
        plt.legend()
        plt.tight_layout()
        if save_this_batch:
            summary_path = rootpath + exp_name + '/summary/'
            plt.savefig(summary_path + f'/Danger.png')
            if save_pdf:
                plt.savefig(summary_path + f'/Danger.pdf')
            sio.savemat(summary_path + '/Danger.mat', all_dangers)
        else:
            plt.show()
        plt.close()

    def _comm_summary():
        plt.figure()
        plt.xlabel('Episode')
        plt.ylabel('Communication Reward')
        colors = plt.colormaps.get_cmap('tab20').colors
        for algo, color in zip(exp_cache['comm_rewards'].keys(), colors):
            plt.plot(exp_cache['comm_rewards'][algo], label=algo, color=color, linewidth=1.5)
        plt.legend()
        plt.tight_layout()
        if save_this_batch:
            summary_path = rootpath + exp_name + '/summary/'
            plt.savefig(summary_path + f'/CommReward.png')
            if save_pdf:
                plt.savefig(summary_path + f'/CommReward.pdf')
            sio.savemat(summary_path + '/Comm.mat', exp_cache['comm_rewards'])
        else:
            plt.show()
        plt.close()

    _reward_summary()
    _danger_summary()
    _comm_summary()
