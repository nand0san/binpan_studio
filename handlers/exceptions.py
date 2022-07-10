import json


############################
# Exceptions
############################

# TODO: color and row exceptions control

class BinanceAPIException(Exception):
    """
    Exceptions from https://github.com/sammchardy/python-binance
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
    Exceptions from https://github.com/sammchardy/python-binance
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f'BinanceRequestException: {self.message}'
