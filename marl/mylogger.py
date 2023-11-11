import logging
from logging.config import fileConfig
from os import path, makedirs

from datetime import datetime

now = datetime.now()


def init_logger(config, log_path=None):
    log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logging_config.ini')
    fileConfig(log_file_path)

    logger = logging.getLogger()
    log_level = config.get('log_level', 'DEBUG')

    log_path = './results/' + now.strftime("%m.%d") + '/PathException/' if log_path is None else log_path
    if not path.exists(log_path):
        makedirs(log_path)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    def _add_file_handler():
        filehandler = logging.FileHandler(log_path + '/log.log', 'a')
        filehandler.setFormatter(formatter)
        filehandler.setLevel('INFO')
        logger.addHandler(filehandler)

    _add_file_handler()

    logger.setLevel(log_level)
    return logger
