import json
from handlers.logs import Logs

exceptions_logger = Logs(filename='./logs/exceptions.log', name='exceptions', info_level='INFO')


############################
# Exceptions
############################

# TODO: color and row exceptions control


class MissingApiData(Exception):

    def __init__(self, message):
        self.message = message
        self.msg = f"""No API Key or API Secret. 
        
        API key would be needed for personal API calls. Any other calls will work.

        Adding:

        from binpan import handlers

        handlers.files.add_api_key("xxxx")
        handlers.files.add_api_secret("xxxx")

        API keys will be added to a file called secret.py in an encrypted way. API keys in memory stay encrypted except in the API call instant.

        Create API keys: https://www.binance.com/en/support/faq/360002502072
        
        Exception: {self.message}
        """

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

    def __str__(self):  # pragma: no cover
        return 'APIError(code=%s): %s' % (self.code, self.message)


class BinanceRequestException(Exception):
    """
    Request methods exceptions.

    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f'BinanceRequestException: {self.message}'


class BinanceOrderException(Exception):
    """
    Orders exceptions from API.

    """

    def __init__(self, code, message):
        self.code = code
        self.message = message

    def __str__(self):
        return 'BinanceOrderException(code=%s): %s' % (self.code, self.message)


class BinanceOrderMinAmountException(BinanceOrderException):
    """
    Symbol filter exceptions: Minimum Amount

    """

    def __init__(self, value):
        message = "Amount must be a multiple of %s" % value
        super().__init__(-1013, message)


class BinanceOrderMinPriceException(BinanceOrderException):
    """
    Symbol filter exceptions: Minimum Price

    """

    def __init__(self, value):
        message = "Price must be at least %s" % value
        super().__init__(-1013, message)


class BinanceOrderMinTotalException(BinanceOrderException):
    """
    Symbol filter exceptions: Minimum Total

    """

    def __init__(self, value):
        message = "Total must be at least %s" % value
        super().__init__(-1013, message)


class BinanceOrderUnknownSymbolException(BinanceOrderException):
    """
    Unknown symbol.

    """

    def __init__(self, value):
        message = "Unknown symbol %s" % value
        super().__init__(-1013, message)


class BinanceOrderInactiveSymbolException(BinanceOrderException):
    """
    Order interactive exception.

    """

    def __init__(self, value):
        message = "Attempting to trade an inactive symbol %s" % value
        super().__init__(-1013, message)


class NotImplementedException(Exception):
    """
    Not implemented.

    """

    def __init__(self, value):
        message = f'Not implemented: {value}'
        super().__init__(message)


class BinPanException(Exception):
    """
    Message the exception and logs it.
    """

    def __init__(self, exception_class, doc, message):
        self.exception_class = exception_class
        self.doc = doc
        self.message = message

    def __str__(self):
        msg = f'BinPan Exception: {self.exception_class} {self.doc} {self.message}'
        exceptions_logger.error(msg)
