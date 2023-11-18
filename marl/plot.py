import matplotlib.pyplot as plt
import numpy as np
import random
import os

from marl import end_rewards, rootpath, folder, save_this_batch, tot_episode
from algo.qmix import losses


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
            plt.savefig(rootpath + batch_name + folder[1] + f'/losses.pdf')
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
            plt.savefig(rootpath + batch_name + folder[1] + f'/EpisodeReward.pdf')
        else:
            pass
            # if episode % (tot_episode // 10) == 0:
            #     plt.show()
        plt.close()

    _rewards()
    _reward_in_one()
    _end_reward()
    _losses()
