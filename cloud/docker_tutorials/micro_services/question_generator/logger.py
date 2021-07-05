from __future__ import absolute_import

import logging
import datetime
import os


def _get_log_file_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = 'amie_core-' + datetime.datetime.now().strftime('%Y-%m-%d') + '.log'
    log_file = os.path.join(base_dir, "runtime_log", file_name)
    log_path = os.path.join(base_dir, "runtime_log")
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    return log_file


def init_logger(log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    log_file = _get_log_file_path()
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_file_level)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger


logger = init_logger()
