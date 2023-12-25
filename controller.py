# -* - coding: UTF-8 -* -
import traceback

import yaml
from marl import *

import os

from tqdm import tqdm

os.environ['NUMEXPR_MAX_THREADS'] = '12'

config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
# TODO add parser


if __name__ == '__main__':
    info = 'simple_tag'  # add 'test' to open the debug mode

    total_batch, seed, episode, seq_len, logger = running_config(config, info)

    # make env
    env = make_env(config['env'])

    for _ in range(total_batch):
        # try:
        one_batch_basic_preparation(config, env)
        # init agent
        CommAgent, TaskAgent = match_agent(config)
        # run experiment
        for e in tqdm(range(episode)):
            results = run_one_episode(seed, env, CommAgent, TaskAgent)
            results = store_results(e, TaskAgent, results, env)
            # plot_results(e, results, config['plot'], env)
            # if e > 0:
            train_agent(CommAgent, TaskAgent)

        batch_summary()



        # except Exception as e:
        #     logger.error(e)
        #     tb = e.__traceback__
        #     traceback.print_tb(tb)

    # exp_summary()
