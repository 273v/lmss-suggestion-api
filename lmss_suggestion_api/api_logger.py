"""lmss_suggestion_api.api_logger module sets up the logging for the application"""

# Copyright (C) 2023 273 Ventures LLC (https://273ventures.com)
# SPDX-License-Identifier: MIT


# system imports
import logging

# project imports
from lmss_suggestion_api.api_settings import ENV

# setup formatter
LOG_FORMAT_STRING = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log_formatter = logging.Formatter(LOG_FORMAT_STRING)

# setup logger
def get_logger(name: str = __name__) -> logging.Logger:
    """
    Setup logger
    :return:
    """
    # setup instance
    logger_instance = logging.getLogger(name)

    # initialize stream handler
    log_handler_stream_instance = logging.StreamHandler()
    log_handler_stream_instance.setFormatter(log_formatter)
    logger_instance.addHandler(log_handler_stream_instance)

    # initialize file handler
    if "LOG_FILE" in ENV and isinstance(ENV["LOG_FILE"], str):
        log_file = ENV["LOG_FILE"]
    else:
        log_file = "lmss_suggestion_api.log"
    log_handler_file_instance = logging.FileHandler(log_file)
    log_handler_file_instance.setFormatter(log_formatter)
    logger_instance.addHandler(log_handler_file_instance)

    # setup debug level
    if ENV["DEBUG"]:
        logger_instance.setLevel(logging.DEBUG)
    else:
        logger_instance.setLevel(logging.WARNING)

    return logger_instance


# main logger
LOGGER = get_logger()
