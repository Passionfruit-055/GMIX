# -* - coding: UTF-8 -* -
import traceback

import yaml
from marl import *

import os
os.environ['NUMEXPR_MAX_THREADS'] = '12'

config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
# TODO add parser

info = 'ActuallyChoose'
logger, batch_name, seed, episode, seq_len = basic_preparation(config, info)

if __name__ == '__main__':
    try:
        # make env
        env = make_env(config['env'])
        # init agent
        CommAgent, GMIXAgent = match_agent(config)
        # run experiment
        for e in range(episode):
            results = run_one_scenario(config, seed, env, CommAgent, GMIXAgent)
            results = store_results(e, batch_name, GMIXAgent, results)
            plot_results(e, batch_name, results, config['plot'])
            # if e > 0:
            train_agent(config, CommAgent, GMIXAgent)

    except Exception as e:
        logger.error(e)
        tb = e.__traceback__
        traceback.print_tb(tb)
