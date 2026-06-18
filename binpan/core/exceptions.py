"""
Exceptions control.

BinPan only defines its own domain exceptions. Errors from the Binance API are
raised by ``panzer`` (``panzer.errors.BinanceAPIException``) and propagate as-is.
"""
from socket import gethostname

from .logs import LogManager

exceptions_logger = LogManager(filename='./logs/exceptions.log', name='exceptions', info_level='INFO')
hostname = gethostname() + ' exceptions '


############################
# Exceptions
############################


class BinPanException(Exception):
    """
    BinPan exception with custom message.

    :param str message: A message for the Exception message.
    """

    def __init__(self, message: str = ''):
        self.message = message
        self.msg = f"BinPan Exception {hostname}: {message}"
        exceptions_logger.error(self.msg)
        super().__init__(self.msg)

    def __str__(self):
        return self.msg


class RedisConfigError(Exception):
    """
    Exception for errors from missing data for redis.
    """

    def __init__(self, message: str = ''):
        self.message = message
        self.msg = f"BinPan Redis Exception {hostname}: {self.message}"
        exceptions_logger.error(self.msg)
        super().__init__(self.msg)

    def __str__(self):
        return self.msg
