import logging
from logging.handlers import RotatingFileHandler

from os import makedirs, path


class Logs:
    def __init__(self, filename='binpan.log', name="BinPan Logger", info_level='DEBUG'):
        try:
            makedirs('logs', exist_ok=True)
        except PermissionError:  # linux
            home = path.expanduser('~')
            makedirs(f'{home}/deploy/deeplab_working', exist_ok=True)
        self.logger = logging.getLogger(name)
        self.log_file = filename

        # Create handlers
        self.screen_handler = logging.StreamHandler()
        # self.file_handler = logging.FileHandler(self.log_file)
        self.file_handler = RotatingFileHandler(self.log_file,
                                                mode='a',
                                                maxBytes=2 * 1024 * 1024,
                                                backupCount=5)

        # Create formatters and add it to handlers
        # self.line_format = '%(asctime)s, %(module)10s %(funcName)20s, %(lineno)5d, %(levelname)8s, %(message)s'
        self.line_format = '%(asctime)s %(levelname)8s %(message)s'
        self.screen_format = logging.Formatter(self.line_format, datefmt='%Y-%m-%d\t %H:%M:%S')
        self.file_format = logging.Formatter(self.line_format, datefmt='%Y-%m-%d\t %H:%M:%S')
        self.screen_handler.setFormatter(self.screen_format)
        self.file_handler.setFormatter(self.file_format)

        # Add handlers to the logger
        self.logger.addHandler(self.screen_handler)
        self.logger.addHandler(self.file_handler)

        # Set level
        self.level = eval(f"logging.{info_level}")
        self.logger.setLevel(self.level)

        # return self.logger

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg, exc_info=True)

    def critical(self, msg):
        self.logger.critical(msg, exc_info=True)
