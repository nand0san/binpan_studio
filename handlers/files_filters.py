from sys import path
from os import path
from .starters import AesCipher
from .logs import Logs

files_logger = Logs(filename='./logs/files_logger.log', name='files_logger', info_level='DEBUG')

cipher_object = AesCipher()


# Interfaces


def read_file(filename: str) -> list:
    """Read a file to a list of strings each line.
    :return: list
    """
    if not path.isfile(filename):
        return []
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


def save_file(filename: str, data: list, mode='w') -> None:
    """Save a new file from a list of lists each line."""
    with open(filename, mode) as f:
        for line in data:
            f.write(str(line) + '\n')


def add_api_key(api_key_value: str) -> None:
    """Checks if exists in a file and if not, then adds a line with the api key value for working with the package."""
    filename = "secret.py"
    saved_data = read_file(filename=filename)
    lines = []
    for line in saved_data:
        if not line.startswith('api_key') and line:
            lines.append(line)
    encryptor = cipher_object.encrypt(api_key_value)
    lines.append(f'\napi_key = "{encryptor}"')
    save_file(filename=filename, data=lines)


def add_api_secret(api_secret_value: str) -> None:
    """Checks if exists in a file and if not, then adds a line with the api secret value for working with the package.
    """
    filename = "secret.py"
    saved_data = read_file(filename=filename)
    lines = [line for line in saved_data if (not line.startswith('api_secret')) and line]
    encryptor = cipher_object.encrypt(api_secret_value)
    lines.append(f'\napi_secret = "{encryptor}"')
    save_file(filename=filename, data=lines)
