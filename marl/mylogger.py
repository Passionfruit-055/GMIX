import logging
from logging.config import fileConfig
from os import path, makedirs

from datetime import datetime
now = datetime.now()


def init_logger():
    log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logging_config.ini')
    fileConfig(log_file_path)
    return logging.getLogger()


def custom_logger(config):
    logger = logging.getLogger()
    log_level = config.get('log_level', 'DEBUG')
    whether_custom_log_path = config.get('custom_file_handler_path', False)
    log_path = './results/' + now.strftime("%m.%d") + '/' if not whether_custom_log_path \
        else config.get('file_handler_path')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S %p')
    if not path.exists(log_path):
        makedirs(log_path)
    filehandler = logging.FileHandler(log_path + 'log.log', 'a')
    filehandler.setFormatter(formatter)
    filehandler.setLevel('INFO')
    logger.addHandler(filehandler)

    logger.setLevel(log_level)
