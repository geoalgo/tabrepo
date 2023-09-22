import logging
import sys


def get_logger():
    logging.basicConfig(stream=sys.stdout, format="%(message)s")
    logger = logging.getLogger("HOexp")
    logger.setLevel(logging.DEBUG)

    return logger
