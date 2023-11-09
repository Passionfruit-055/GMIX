import traceback
import logging

import numpy as np
import yaml
from marl import make_env, match_agent, run_one_scenario, store_results, plot_results, set_seed, train_agent
from marl.mylogger import custom_logger

# load config from yaml file
config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
custom_logger(config['experiment']['logger'])
logger = logging.getLogger()

try:
    seed = set_seed(config['experiment']['running'].get('seed', 21))
    # make env
    env = make_env(config['env'])
    # init agent
    CommAgent, GMIXAgent = match_agent(config)

except Exception as e:
    logger.error(e)
    tb = e.__traceback__
    traceback.print_tb(tb)

episode = config['experiment']['running'].get('episode', 1000)
timestep = config['experiment']['running'].get('timestep', 100)

for e in range(episode):
    results = run_one_scenario(config, seed, env, CommAgent, GMIXAgent)
    store_results(episode=e, **results)
    plot_results(episode=e, **results)
    train_agent(config, CommAgent, GMIXAgent)


def func(**kwargs):
    for k, v in kwargs.items():
        print(k, v)


if '__main__' == __name__:
    kw = {'a': 1, 'b': 2, 'c': 3}
    func(**kw)
