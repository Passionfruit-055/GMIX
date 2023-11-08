import traceback
import logging

import yaml
from marl import make_env, match_agent
from marl.mylogger import custom_logger

# load config from yaml file
config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
custom_logger(config['experiment']['logger'])
logger = logging.getLogger()

try:
    # make env
    env = make_env(config['env'])
    # init agent
    CommAgent, GMIXAgent = match_agent(config)

except Exception as e:
    logger.error(e)
    tb = e.__traceback__
    traceback.print_tb(tb)

episode = config.get('episode', 1000)
timestep = config.get('timestep', 100)

# results =
