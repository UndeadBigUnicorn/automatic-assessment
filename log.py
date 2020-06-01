import logging

logging.basicConfig(format='%(levelname)s %(asctime)s - %(message)s', level=logging.INFO)


def debug(msg, *args, **kwargs):
    logging.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    logging.info(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    logging.error(msg, *args, **kwargs)
