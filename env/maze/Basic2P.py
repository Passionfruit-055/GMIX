import random
from copy import deepcopy

import numpy as np

from env.maze.maze import Maze2D
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"  # can assign any character,
import pygame


class Player(object):
    def __init__(self, index, basemap, name='default', lives=10, communicate=False, view=2):
        self.index = index
        self.name = name
        self.max_life = lives
        self.cur_life = lives
        self.communicate = communicate
        self.trap = 0
        self.bonus = 0
        self.succeed = False

        self.moves = []  # agent defines
        self.path = []  # actual path according to map
        self.next_stop = None
        self.map = basemap

        self.view = view

    def load(self, starting_point, end_point):
        self.cur_life = self.max_life
        self.trap = 0
        self.bonus = 0
        self.moves = []
        self.pos = starting_point
        self.goal = end_point
        # print(f"{self.name} start at {self.pos} goal at {self.goal}")
        self.succeed = False
        self.next_stop = None

    def cur_status(self, cmd=False):

        if cmd:
            print(f"{self.name} At ({self.pos[0]}, {self.pos[1]})")
            print(f"Life: {self.cur_life}")
            print(f"Score: {round(self.score, 2)}")
            print(f"Traps: {self.trap}\n")

        return {self.name: {'pos': self.pos, 'life': self.cur_life, 'trap': self.trap, 'bonus': self.bonus, }}

    def end_status(self):
        if self.cur_life <= 0:
            status = 'Dead'
        elif self.succeed:
            status = 'Succeed'
        else:
            status = 'Failed'
        # print(f"{self.name}: {status}")
        # print(f"Score: {round(self.score, 3)}")
        # print(f"Keep alive for {len(self.moves)} steps")
        return status

    @property
    def score(self):
        life = 1.2
        bonus = 1
        trap = 0.5
        time = 0.5
        basicScore = life * self.cur_life + bonus * self.bonus - trap * self.trap - time * len(self.moves)
        if self.cur_life <= 0:  # 死亡惩罚
            basicScore -= 5
        return basicScore

    def detect(self, action):
        # 判断地图是否允许本次移动
        # 'up': 0, 'down': 1, 'left': 2, 'right': 3, 'stay': 4

        self.next_stop = self.pos
        reachable = True
        self.moves.append(action)

        x, y = self.pos
        if action == 0:
            dest = (x, y - 1)
        elif action == 1:
            dest = (x, y + 1)
        elif action == 2:
            dest = (x - 1, y)
        elif action == 3:
            dest = (x + 1, y)
        else:
            raise ValueError(f"Invalid action {action}")

        if dest[0] < 0 or dest[0] >= self.map.shape[0] or dest[1] < 0 or dest[1] >= self.map.shape[1]:
            reachable = False
        else:
            self.next_stop = dest

        return self.next_stop, reachable

    def observe(self):
        state = []
        view = self.view
        for i in range(self.pos[0] - view, self.pos[0] + view + 1):
            for j in range(self.pos[1] - view, self.pos[1] + view + 1):
                if 0 <= i <= self.map.shape[0] - 1 and 0 <= j <= self.map.shape[1] - 1:
                    state.append(self.map[i][j])
                else:
                    state.append(0)
        return state


class Basic2P(Maze2D):
    def __init__(self, width=6, height=6, grid_size=50, risk_level=0.1, bonus_level=0.1, players=2, render=True, FPS=60,
                 timestep=50, player_lives=100):

        super().__init__(width, height, grid_size, risk_level=risk_level, bonus_level=bonus_level)

        self.players = [Player(i, self.map, name=f'Player{i}', lives=player_lives) for i in
                        range(players)]
        self.remaining_bonuses = [tuple(bonus.copy()) for bonus in self.bonuses]
        self.name = 'Basic2P'

        self.timestep = timestep
        self.countdown = timestep

        self.ACT = ['up', 'down', 'left', 'right', 'stay']

        self.running = True

        # render
        self.render_mode = render
        self.screen = None
        self.clock = None
        self.FPS = FPS

    def step(self, actions=None):

        self.countdown -= 1

        actions = list(actions.values()) if isinstance(actions, dict) else actions

        self._player_move(actions)

        rewards = self.rewards[0: 2]

        observations = self.observe()

        terminations = {player: False if self.countdown else True for player in self.players}

        truncations = {player: True if player.cur_life <= 0 or player.succeed else False for player in self.players}

        self.running = False if all(terminations.values()) is True or all(truncations.values()) is True else True



        return observations, rewards, terminations, truncations, self.infos

    def demo(self):
        self.reset()
        end_time = self.timestep

        for t in range(self.timestep):

            movable = self._player_move()

            if self.render_mode:
                self._render()
                self.clock.tick(self.FPS)

            if movable == 0:
                end_time = t + 1
                if self.render_mode:
                    pygame.quit()
                break

        return self.episode_analysis(end_time)

    def _player_move(self, actions=None):
        dests = []
        movable_player = []
        actions = [random.choice(np.arange(0, 4, 1).tolist()) for _ in
                   range(self.agent_num)] if actions is None else actions
        for player, action in zip(self.players, actions):
            if player.cur_life <= 0 or player.succeed:
                pass
            else:
                dest, reachable = player.detect(action)
                dests.append(dest)
                movable_player.append(player)

        def dest_conflict(arr):
            indices_dict = {}
            seen = list()
            for i in range(len(arr)):
                element = arr[i]
                if element not in seen:
                    seen.append(element)
                else:
                    if element in indices_dict:
                        indices_dict[element].append(i)
                    else:
                        indices_dict[element] = [i]
            # 产生冲突时，序号小的优先级更高
            for key in indices_dict.keys():
                indices_dict[key] = indices_dict[key][1:]
            # 最后只需要那些无法行动的玩家的序号
            return indices_dict.values()

        # achieve parallel
        conflict_players = dest_conflict(dests)

        for i, player in enumerate(movable_player):
            if i in conflict_players:
                player.cur_life -= 0.1  # 无论是撞墙还是产生冲突导致不可达，都要扣分
                player.path.append(4)
            else:
                if player.next_stop == player.goal:
                    player.bonus += 10
                    player.succeed = True
                elif player.next_stop in self.remaining_bonuses:
                    player.bonus += 1
                    self.remaining_bonuses.remove(player.next_stop)
                elif player.next_stop in self.dangers:
                    player.cur_life -= 1
                    player.trap += 1
                player.path.append(player.moves[-1])
                player.pos = player.next_stop

        return len(movable_player)

    def reset(self, seed=0):
        self.running = True

        self.countdown = self.timestep

        for (player, sp, ep) in zip(self.players, self.starting_points, self.ending_points[::-1]):
            player.load(sp, ep)

        self.remaining_bonuses = [tuple(bonus) for bonus in self.bonuses]

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width * self.grid_size, self.height * self.grid_size))
            pygame.display.set_caption(self.name)
            self.clock = pygame.time.Clock()

        return self.observe(), self.infos

    def _render(self):
        # load texture
        self.icons = {'player': pygame.image.load('assets/agent.png'),
                      'danger': pygame.image.load('assets/danger.png'),
                      'bonus': pygame.image.load('assets/cargo.png'),
                      'goal': pygame.image.load('assets/dest.png')}

        self.screen.fill('#FFFEF7')

        self._render_entity()

        for i in range(self.height):
            pygame.draw.line(self.screen, 'black', (0, i * self.grid_size),
                             (self.width * self.grid_size, i * self.grid_size))
        for j in range(self.width):
            pygame.draw.line(self.screen, 'black', (j * self.grid_size, 0),
                             (j * self.grid_size, self.grid_size * self.height))

        pygame.display.update()
        pygame.time.delay(200)

    def _render_entity(self):

        for danger in self.dangers:
            self.screen.blit(self.icons['danger'], (danger[0] * self.grid_size + 3, danger[1] * self.grid_size + 2))

        for bonus in self.remaining_bonuses:
            self.screen.blit(self.icons['bonus'], (bonus[0] * self.grid_size, bonus[1] * self.grid_size))

        for goal in self.ending_points:
            self.screen.blit(self.icons['goal'], (goal[0] * self.grid_size + 3, goal[1] * self.grid_size + 2))

        for player in self.players:
            self.screen.blit(self.icons['player'], (player.pos[0] * self.grid_size, player.pos[1] * self.grid_size))

    def episode_analysis(self, end_time):
        # print('\n' + '*' * 20 + '\n')
        difficulty = 'Learnable'
        info = {'dead': 0, 'succeed': 0, 'failed': 0}
        for player in self.players:
            status = player.end_status()
            info[status.lower()] += 1

        if info['dead'] == len(self.players):
            difficulty = 'Dangerous'
        elif info['succeed'] == len(self.players):
            difficulty = 'Easy'
        elif info['failed'] == len(self.players) or (
                info['failed'] + info['dead'] == len(self.players) and info['failed'] != 0):
            difficulty = 'Short'
        elif info['succeed'] + info['failed'] == len(self.players):
            difficulty = 'Hard'
        return difficulty, end_time

    @property
    def agent_num(self):
        return len(self.players)

    @property
    def obs_space(self):
        return (2 * self.players[0].view + 1) ** 2 + self.agent_num  # onehot

    @property
    def action_space(self):
        return len(self.ACT) - 1

    @property
    def state_space(self):
        return self.obs_space * self.agent_num

    @property
    def agents(self):
        return [player.name for player in self.players if player.cur_life > 0 and self.running]

    def observe(self):
        obs = []
        onehots = np.eye(self.agent_num)
        for player, onehot in zip(self.players, onehots):
            obs.append(player.observe() + list(onehot))
        return np.array(obs, dtype=np.float32)

    @property
    def rewards(self):
        rewards = [player.score for player in self.players]
        reward_all = np.sum(rewards)
        rewards.append(reward_all)
        return rewards

    @property
    def danger_times(self):
        return [player.trap for player in self.players]

    @property
    def infos(self):
        return {'traps': [player.trap for player in self.players],
                'bonus': [player.bonus for player in self.players],
                'score': [player.score for player in self.players],
                'status': [player.cur_status() for player in self.players]}


if __name__ == '__main__':
    game = Basic2P(6, 6, 50, 0.1, 0.1, 2)
    game.demo()
