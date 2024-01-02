import os
import numpy as np
import random

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"  # can assign any character,
import pygame


class Maze2D(object):
    def __init__(self, width=6, height=6, grid_size=50, sp_seq=None, ep_seq=None, dp_seq=None, risk_level=0.1,
                 bp_seq=None, bonus_level=0.1):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.map = np.zeros((width, height))

        self.starting_points = sp_seq if sp_seq is not None else [[0, 0], [0, self.height - 1]]  # 左上右上

        self._assign(self.starting_points, -2)
        self.ending_points = ep_seq if ep_seq is not None else [[self.width - 1, 0],
                                                                [self.width - 1, self.height - 1]]  # 左下右下

        self._assign(self.ending_points, 10)

        self.dangers = dp_seq if dp_seq is not None else self._distribute(risk_level)
        self._assign(self.dangers, -1)

        self.bonuses = bp_seq if bp_seq is not None else self._distribute(bonus_level)
        self._assign(self.bonuses, 2)

    def _distribute(self, dense_level=0.2):
        num = round(self.width * self.height * dense_level)
        # assert num > 0, f"Distribution num should be larger than zero"

        def random_nonzero_coordinates(arr, n):
            # 找到所有零元素的索引
            zero_indices = np.argwhere(arr == 0)
            # 随机选择n个非零元素的索引
            selected_indices = np.random.choice(len(zero_indices), size=n, replace=False)

            return zero_indices[selected_indices]

        return random_nonzero_coordinates(self.map, num)

    def _assign(self, seq, value):
        for coord in seq:
            width, height = coord
            self.map[width][height] = value

    def show(self):
        pygame.init()
        screen = pygame.display.set_mode((self.width * self.grid_size, self.height * self.grid_size))
        pygame.display.set_caption("Maze2D")
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            screen.fill((241, 240, 237))

            for i in range(self.width):
                for j in range(self.height):
                    if self.map[i][j] == -1:
                        pygame.draw.rect(screen, (238, 44, 121),
                                         (i * self.grid_size + 5, j * self.grid_size + 5, self.grid_size - 10,
                                          self.grid_size - 10))
                    elif self.map[i][j] == -2:
                        pygame.draw.rect(screen, (27, 167, 132),
                                         (i * self.grid_size + 10, j * self.grid_size + 10, self.grid_size - 20,
                                          self.grid_size - 20))
                    elif self.map[i][j] == 2:
                        pygame.draw.rect(screen, (254, 215, 26),
                                         (i * self.grid_size, j * self.grid_size, self.grid_size,
                                          self.grid_size))
                    elif self.map[i][j] == 10:
                        pygame.draw.circle(screen, (23, 129, 181), (
                            i * self.grid_size + self.grid_size / 2, j * self.grid_size + self.grid_size / 2),
                                           self.grid_size // 2 - 2)
            for i in range(self.height):
                pygame.draw.line(screen, 'black', (0, i * self.grid_size),
                                 (self.width * self.grid_size, i * self.grid_size))
            for j in range(self.width):
                pygame.draw.line(screen, 'black', (j * self.grid_size, 0),
                                 (j * self.grid_size, self.grid_size * self.height))

            pygame.display.update()


if __name__ == '__main__':
    maze = Maze2D()
    maze.show()
