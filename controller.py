# -* - coding: UTF-8 -* -
import traceback

import yaml
from marl import *

import os

os.environ['NUMEXPR_MAX_THREADS'] = '12'

config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
# TODO add parser


if __name__ == '__main__':
    info = 'testCommStore'  # inject 'test' to open the debug mode

    total_batch, seed, episode, seq_len, logger = running_config(config, info)

    # make env
    env = make_env(config['env'])

    for _ in range(total_batch):

        try:
            batch_name = one_batch_basic_preparation(config, env)
            # init agent
            CommAgent, TaskAgent = match_agent(config)
            # run experiment
            for e in range(episode):
                results = run_one_episode(config, seed, env, CommAgent, TaskAgent)
                results = store_results(e, batch_name, TaskAgent, results)
                plot_results(e, batch_name, results, config['plot'])
                # if e > 0:
                train_agent(CommAgent, TaskAgent)

        except Exception as e:
            logger.error(e)
            tb = e.__traceback__
            traceback.print_tb(tb)

    exp_summary()
