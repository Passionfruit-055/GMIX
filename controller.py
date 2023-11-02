import yaml
import marltools
from marltools import logger, make_env, build_model
from marltools.mylogger import custom_logger

# load config from yaml file
config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
exp_config, model_config, env_config = config.values()
custom_logger(exp_config['logger'])

# make env
try:
    env, env_config = make_env(env_config)
except Exception as e:
    logger.error(e)

# make model
model_config.update(env_config)

try:
    model, model_config = build_model(model_config)
except Exception as e:
    logger.error(e)