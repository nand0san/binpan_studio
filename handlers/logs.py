"""

Logging functions

"""

import logging
from os import makedirs, path
from logging.handlers import RotatingFileHandler


class Logs:
    """
    Main logging class.
    """
    def __init__(self, filename='binance_cache.log', name="binance_cache", info_level='DEBUG', max_log_size_mb=10, backup_count=5):
        try:
            makedirs('logs', exist_ok=True)
        except PermissionError:  # linux
            home = path.expanduser('~')
            makedirs(f'{home}/deploy/deeplab_working', exist_ok=True)

        self.logger = logging.getLogger(name)

        # avoid duplicated logs
        if self.logger.hasHandlers():
            self.logger.handlers = []

        self.log_file = filename

        # Create handlers
        self.screen_handler = logging.StreamHandler()
        # self.file_handler = logging.FileHandler(self.log_file)
        self.file_handler = RotatingFileHandler(self.log_file, maxBytes=max_log_size_mb * 1024 * 1024, backupCount=backup_count, encoding='utf-8')

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

    def debug(self, msg):
        """DEBUG Method"""
        self.logger.debug(msg)

    def info(self, msg):
        """INFO Method"""
        self.logger.info(msg)

    def warning(self, msg):
        """WARNING Method"""
        self.logger.warning(msg)

    def error(self, msg):
        """ERROR Method"""
        self.logger.error(msg, exc_info=True)

    def critical(self, msg):
        """CRITICAL Method"""
        self.logger.critical(msg, exc_info=True)

    def read_last_lines(self, num_lines):
        with open(self.log_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            return lines[-num_lines:]

