"""
Exceptions control.
"""
import json
from socket import gethostname

from .logs import Logs

exceptions_logger = Logs(filename='./logs/exceptions.log', name='exceptions', info_level='INFO')
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


class MissingBinanceApiData(Exception):
    """
    Exception for errors from missing data in an API request.
    """

    def __init__(self, message: str = ''):
        self.message = message
        self.msg = f"BinPan Binance API Key or API Secret Exception {hostname}: {self.message}"
        exceptions_logger.error(self.msg)
        super().__init__(self.msg)

    def __str__(self):
        return self.msg


class MissingTelegramApiData(Exception):
    """
    Exception for errors from missing data for telegram.
    """

    def __init__(self, message: str = ''):
        self.message = message
        self.msg = f"BinPan Telegram Exception {hostname}: {self.message}"
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


class BinanceAPIException(Exception):
    """
    Extract exception from response.

    """

    def __init__(self, response, status_code, text):
        self.code = 0
        try:
            json_res = json.loads(text)
        except ValueError:
            self.message = 'Invalid JSON error message from Binance: {}'.format(response.text)
        else:
            self.code = json_res['code']
            self.message = json_res['msg']
        self.status_code = status_code
        self.response = response
        self.request = getattr(response, 'request', None)
        exceptions_logger.error(self.message)

    def __str__(self):  # pragma: no cover
        return 'APIError(code=%s): %s' % (self.code, self.message)


class BinanceRequestException(Exception):
    """
    Request methods exceptions.

    """

    def __init__(self, message):
        self.message = message
        exceptions_logger.error(self.message)

    def __str__(self):
        return f'BinanceRequestException: {self.message}'


class BinanceOrderException(Exception):
    """
    Orders exceptions from API.

    """

    def __init__(self, code, message):
        self.code = code
        self.message = message
        exceptions_logger.error(self.message)

    def __str__(self):
        return 'BinanceOrderException(code=%s): %s' % (self.code, self.message)


class BinanceOrderMinAmountException(BinanceOrderException):
    """
    Symbol filter exceptions: Minimum Amount

    """

    def __init__(self, value):
        message = "Amount must be a multiple of %s" % value
        exceptions_logger.error(self.message)
        super().__init__(-1013, message)


class BinanceOrderMinPriceException(BinanceOrderException):
    """
    Symbol filter exceptions: Minimum Price

    """

    def __init__(self, value):
        message = "Price must be at least %s" % value
        exceptions_logger.error(self.message)
        super().__init__(-1013, message)


class BinanceOrderMinTotalException(BinanceOrderException):
    """
    Symbol filter exceptions: Minimum Total

    """

    def __init__(self, value):
        message = "Total must be at least %s" % value
        exceptions_logger.error(self.message)
        super().__init__(-1013, message)


class BinanceOrderUnknownSymbolException(BinanceOrderException):
    """
    Unknown symbol.

    """

    def __init__(self, value):
        message = "Unknown symbol %s" % value
        exceptions_logger.error(self.message)
        super().__init__(-1013, message)


class BinanceOrderInactiveSymbolException(BinanceOrderException):
    """
    Order interactive exception.

    """

    def __init__(self, value):
        message = "Attempting to trade an inactive symbol %s" % value
        exceptions_logger.error(self.message)
        super().__init__(-1013, message)


class NotImplementedException(Exception):
    """
    Not implemented.

    """

    def __init__(self, value):
        self.message = f'Not implemented: {value}'
        exceptions_logger.error(self.message)
        super().__init__(self.message)
