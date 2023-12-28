import random

import numpy as np

from map import Maze2D
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"  # can assign any character,
import pygame


class Player(object):
    def __init__(self, index, basemap, pos, name='default', lives=6, communicate=False):
        self.index = index
        self.name = name
        self.max_life = lives
        self.cur_life = lives
        self.communicate = communicate
        self.trap = 0
        self.bonus = 0
        self.succeed = False

        self.moves = []
        self.pos = pos
        self.map = basemap

    def load(self, starting_point, end_point):
        self.cur_life = self.max_life
        self.trap = 0
        self.bonus = 0
        self.moves = []
        self.pos = starting_point
        self.goal = end_point
        self.succeed = False

    def show_status(self, show_name=False):
        name = self.name if show_name else f"player{self.index}"
        # print(f"Status of {name}:")
        print(f"At ({self.pos[0]}, {self.pos[1]})")
        print(f"Life: {self.cur_life}")
        print(f"Score: {self.score}")
        print(f"Traps: {self.trap}\n")

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
        dest = self.pos
        reachable = True
        # TODO 智能体位置重合判定
        if self.cur_life >= 0 and not self.succeed:
            # 'up': 0, 'down': 1, 'left': 2, 'right': 3, 'stay': 4
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

        return dest, reachable

    def observe(self, view=2):
        state = []
        for i in range(self.pos[0] - view, self.pos[0] + view):
            for j in range(self.pos[1] - view, self.pos[1] + view):
                if 0 <= i <= self.map.shape[0] - 1 and 0 <= j <= self.map.shape[1] - 1:
                    state.append(self.map[i][j])
                else:
                    state.append(0)
        return state


class Basic2P(Maze2D):
    def __init__(self, width=6, height=6, grid_size=50, risk_level=0.1, bonus_level=0.1, players=2, render=True, FPS=60,
                 timestep=25):

        super().__init__(width, height, grid_size, risk_level=risk_level, bonus_level=bonus_level)

        self.players = [Player(i, self.map, self.starting_points[i], name=f'Player{i}') for i in range(players)]
        self.render_mode = render
        self.name = 'Basic2P'
        self.timestep = timestep
        self.FPS = FPS
        self.ACT = ['up', 'down', 'left', 'right', 'stay']

    def run(self):
        self._load_game()
        for t in range(self.timestep):

            self.step()

            if self.render_mode:
                self._render()
                self.clock.tick(self.FPS)

            self.store()

    def step(self):
        for player in self.players:
            action = random.choice(np.arange(0, 4, 1).tolist())
            print(f"{player.name} goes {self.ACT[action]}")
            dest, reachable = player.detect(action)
            if reachable:
                if dest == player.goal:
                    player.bonus += 10
                    player.succeed = True
                elif dest in self.bonuses:
                    player.bonus += 1
                elif dest in self.dangers:
                    player.cur_life -= 1
                    player.trap += 1
                player.pos = dest
                player.moves.append(action)
            else:
                player.cur_life -= 0.5
                player.moves.append(4)
            player.show_status()

    def _load_game(self):

        for (player, sp, ep) in zip(self.players, self.starting_points, self.starting_points[::-1]):
            player.load(sp, ep)

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width * self.grid_size, self.height * self.grid_size))
            pygame.display.set_caption(self.name)
            self.clock = pygame.time.Clock()

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
        pygame.time.delay(1000)

    def _render_entity(self):

        for danger in self.dangers:
            self.screen.blit(self.icons['danger'], (danger[0] * self.grid_size + 3, danger[1] * self.grid_size + 2))
        # TODO bonus 被拿走后要不要存在，感觉从地图不变化的角度，不应该清零
        for bonus in self.bonuses:
            self.screen.blit(self.icons['bonus'], (bonus[0] * self.grid_size, bonus[1] * self.grid_size))

        for goal in self.ending_points:
            self.screen.blit(self.icons['goal'], (goal[0] * self.grid_size + 3, goal[1] * self.grid_size + 2))

        for player in self.players:
            self.screen.blit(self.icons['player'], (player.pos[0] * self.grid_size, player.pos[1] * self.grid_size))
    def store(self):
        pass


if __name__ == '__main__':
    game = Basic2P(6, 6, 50, 0.1, 0.1, 2)
    game.run()
