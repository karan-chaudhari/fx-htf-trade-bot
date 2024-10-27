import os
import logging
from dotenv import load_dotenv
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, date as dt

# Loads .env variables
load_dotenv()

def setup_logger(logger_name, logfile):
    """
    Function to initialize the logger, notice that it takes 2 arguments
    logger_name and logfile
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # create a TimedRotatingFileHandler which logs even debug messages
    fh = TimedRotatingFileHandler(logfile, when="midnight", interval=1)
    fh.setLevel(logging.INFO)
    fh.suffix = "%Y-%m-%d"  # Add a suffix to the log file with the date
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    # formatter = logging.Formatter('%(asctime)s - %(levelname)-6s - call_trace=%(pathname)s L%(lineno)-4d - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(levelname)-6s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    # Stop propagation
    logger.propagate = False
    return logger

######## LOGGER CONFIGURATION ########
date_today = dt.today()

LOG_PATH = "logs/"

## APP LOGGER
logger = setup_logger('app_logger', f"{LOG_PATH}fx_bot.log")
