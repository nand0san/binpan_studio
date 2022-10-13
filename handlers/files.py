import pandas as pd
from csv import QUOTE_ALL
from time import time
from sys import path
from os import listdir, path, mkdir, replace, makedirs

from .starters import AesCipher
from .logs import Logs

files_logger = Logs(filename='./logs/files_logger.log', name='files_logger', info_level='INFO')

cipher_object = AesCipher()


################
# Files manage #
################

def create_dir(my_path: str):
    """
    Create a folder at the designated path if it doesn't exist. No error if it already exists

    :param str my_path: Path for the new directory.

    """
    if not path.exists(my_path):
        makedirs(my_path)


def save_dataframe_to_csv(filename, data: pd.DataFrame, col_sep=',', index=False, timestamp=True) -> None:
    """
    Save a dataframe in a csv with a separator between columns. Each field of each column will be enclosed in double quotes.

    :param str filename: The file name.
    :param pd.DataFrame data: The dataframe with date in columns.
    :param str col_sep: Default is ','
    :param bool index: Keep index or not when saving. DEfault is False (drop index)
    :param bool timestamp: Add a timestamp to the file name automatically. Default is true.

    """
    if filename.lower().endswith(".csv"):
        filename = filename.replace('.csv', '')
    if timestamp:
        filename = filename + '_' + str(time()).split('.')[0]
    filename += '.csv'
    data.to_csv(filename, sep=col_sep, header=True, encoding='utf-8', quoting=QUOTE_ALL, index=index)


def append_dataframe_to_csv(filename: str, data: pd.DataFrame, col_sep: str = ',', index: bool = False, header=True) -> None:
    """
    Add lines to a csv with separator to choose, a dataframe. Each field of each column will be enclosed in quotes

    :param str filename: The file to use.
    :param pd.DataFrame data: Data to append to the file.
    :param str col_sep: Expected column separators. Default is ","
    :param bool index: Keeps index in file as first column/s.
    :param bool header: Keeps header. Default is True.
    """
    if path.isfile(filename):
        header = False
    if filename.lower().endswith(".csv"):
        filename = filename.replace('.csv', '')
    filename += '.csv'
    data.to_csv(filename, sep=col_sep, header=header, encoding='utf-8', quoting=QUOTE_ALL, index=index, mode='a')


def find_csvs_in_path(files_path: str = '.', extension='csv'):
    """
    Locate all files with desired extension.

    :param str files_path: A path.
    :param str extension: A extension like "csv".
    :return:
    """
    ret = []
    for r in listdir(files_path):
        full_path = path.join(files_path, r)
        if path.isfile(full_path):
            ret.append(full_path)
    return [f for f in ret if f.endswith(extension)]


def move_old_csvs(files_path: str = '.', extension='csv'):
    """
    Move all existing files in a path with desired extension to a folder called "old".

    :param str files_path: A path.
    :param str extension: A extension. Default is "csv".

    """
    files_path = path.abspath(files_path)
    old = path.join(files_path, 'old')
    if not path.exists(old):
        mkdir(old)
    csvs = find_csvs_in_path(files_path=files_path, extension=extension)
    for file in csvs:
        file_name = path.basename(file)
        dst = path.join(files_path, 'old', file_name)
        print(f'Moving old file {file}')
        replace(file, dst)


def read_csv_to_dataframe(filename: str, col_sep: str = ',', index_col: bool = None) -> pd.DataFrame:
    """
    Creates a csv file from a dataframe.

    :param str filename: The file name with or without path.
    :param str col_sep: Column separator. DEfault is ","
    :param bool index_col: If False, index is dropped. Default is False.
    :return pd.DataFrame: A dataframe with data in columns using file rows header.
    """
    return pd.read_csv(filepath_or_buffer=filename, sep=col_sep, index_col=index_col, skip_blank_lines=True,
                       quoting=QUOTE_ALL)


def read_file(filename: str) -> list:
    """
    Read a file to a list of strings each line.

    :return list: list with a string each row in the file.

    """
    if not path.isfile(filename):
        return []
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


def save_file(filename: str, data: list, mode='w') -> None:
    """
    Save a new file from a list of lists each line.

    :param str filename: a file name to save.
    :param list data: Data in a list of strings each line.
    :param str mode: 'w' to rewrite full file or 'a' to append to existing file.

    """
    with open(filename, mode) as f:
        for line in data:
            f.write(str(line) + '\n')

###################
# API AND SECRETS #
###################


def add_api_key(api_key_value: str) -> None:
    """
    Checks if exists in secret.py file and if not, then adds a line with the API key (not secret API key) encrypted value for working
    with the package.

    :param str api_key_value: API key
    """
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
    """
    Checks if exists in secret.py file and if not, then adds a line with the API secret encrypted value for working with the package.

    :param str api_secret_value: API key
    """
    filename = "secret.py"
    saved_data = read_file(filename=filename)
    lines = [line for line in saved_data if (not line.startswith('api_secret')) and line]
    encryptor = cipher_object.encrypt(api_secret_value)
    lines.append(f'\napi_secret = "{encryptor}"')
    save_file(filename=filename, data=lines)


def add_any_key(key: str, key_name: str) -> None:
    """
    Checks if exists in a file and if not, then adds a line with the api key value for working with the package.

    :param str key: Any API key or secret to encrypt in secret.py file.
    :param str key_name: Variable name to import it later.

    Example to add telegram data for notifications handler:

        .. code-block:: python

            from binpan import handlers

            handlers.add_any_key(key="xxxxxxxxxx", key_name="encoded_chat_id")
            handlers.add_any_key(key="xxxxxxxxxx", key_name="encoded_telegram_bot_id")


    """
    filename = "secret.py"
    saved_data = read_file(filename=filename)
    lines = []
    for line in saved_data:
        if line:
            if not line.startswith(key_name):
                lines.append(line)
    encryptor = cipher_object.encrypt(key)
    lines.append(f'\n{key_name} = "{encryptor}"')
    save_file(filename=filename, data=lines)
