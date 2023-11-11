import traceback

import yaml
from marl import *

# load config from yaml file
config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

info = 'test to mat'
logger, batch_name, seed, episode, seq_len = basic_preparation(config, info)

try:
    # make env
    env = make_env(config['env'])
    # init agent
    CommAgent, GMIXAgent = match_agent(config)

except Exception as e:
    logger.error(e)
    tb = e.__traceback__
    traceback.print_tb(tb)

for e in range(episode):
    results = run_one_scenario(config, seed, env, CommAgent, GMIXAgent)
    results = store_results(e, batch_name, GMIXAgent, results)
    plot_results(e, results, config['plot'])
    train_agent(config, CommAgent, GMIXAgent)


def func(**kwargs):
    for k, v in kwargs.items():
        print(k, v)


if '__main__' == __name__:
    kw = {'a': 1, 'b': 2, 'c': 3}
    func(**kw)
