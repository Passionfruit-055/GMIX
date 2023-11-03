import traceback

import yaml
import marl
from marl import logger, make_env, build_model
from marl.mylogger import custom_logger

# load config from yaml file
config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
exp_config, model_config, env_config = config.values()
custom_logger(exp_config['logger'])


try:
    # make env
    env, env_config = make_env(env_config)
    model_config.update(env_config)
    # make model
    model, model_config = build_model(model_config)

except Exception as e:
    logger.error(e)
    tb = e.__traceback__
    traceback.print_tb(tb)




