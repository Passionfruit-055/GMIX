import os

from Basic2P import Basic2P, Player
from maze import Maze2D
import numpy as np
import csv
import tqdm
from datetime import datetime

now = datetime.now()
now = now.strftime("%m-%d-%H-%M")

# test param
player_lives = np.arange(50, 101).tolist()
map_size = np.arange(4, 11).tolist()
risk_level = np.arange(0, 0.4, 0.1).tolist()
bonus_level = np.arange(0.1, 0.4, 0.1).tolist()
timestep = np.arange(50, 101, 1).tolist()
average_time = 10
Difficult = {'Easy': 1, 'Short': 2, 'Learnable': 3, 'Hard': 4, 'Dangerous': 5}

info = 'changeScoreCondition'

save_path = f'./test/{now}-{info}.csv'
if not os.path.exists(save_path):
    open(save_path, 'w').close()

with open(save_path, 'a+', newline='') as file:
    writer = csv.writer(file)
    writer.writerows([['player_lives', 'map_size', 'risk_level', 'bonus_level', 'timestep', 'difficulty']])

for p in player_lives:
    for m in map_size:
        for r in risk_level:
            for b in bonus_level:
                for t in timestep:
                    hard_index = 0
                    for i in range(average_time):
                        env = Basic2P(width=m, height=m, risk_level=r, bonus_level=b, player_lives=p, render=False,
                                      timestep=t)
                        difficulty, end_time = env.run()
                        hard_index += Difficult[difficulty]
                    hard_index /= average_time
                    with open(save_path, 'a+', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows([[p, m, r, b, t, hard_index]])
