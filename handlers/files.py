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

def create_dir(path):
    """Crea una carpeta en la ruta designada si no existe. No da error si ya existe"""
    if not path.exists(path):
        makedirs(path)


def save_dataframe_to_csv(filename, data: pd.DataFrame, col_sep=',', index=False, timestamp=True) -> None:
    """Graba en un csv con separador a elegir, un dataframe. Cada campo de cada columna irá entrecomillado"""
    if filename.lower().endswith(".csv"):
        filename = filename.replace('.csv', '')
    if timestamp:
        filename = filename + '_' + str(time()).split('.')[0]
    filename += '.csv'
    data.to_csv(filename, sep=col_sep, header=True, encoding='utf-8', quoting=QUOTE_ALL, index=index)


def append_row_to_csv(filename, data: pd.DataFrame, col_sep=',', index=False, timestamp=True, header=True) -> None:
    """Añade lineas a un csv con separador a elegir, un dataframe. Cada campo de cada columna irá entrecomillado"""
    if path.isfile(filename):
        header = False
    if filename.lower().endswith(".csv"):
        filename = filename.replace('.csv', '')
    if timestamp:
        filename = filename + '_' + str(time()).split('.')[0]
    filename += '.csv'
    data.to_csv(filename, sep=col_sep, header=header, encoding='utf-8', quoting=QUOTE_ALL, index=index, mode='a')


def find_csvs_in_path(files_path: str = '.', extension='csv'):
    ret = []
    for r in listdir(files_path):
        full_path = path.join(files_path, r)
        if path.isfile(full_path):
            ret.append(full_path)
    return [f for f in ret if f.endswith(extension)]


def move_old_csvs(files_path: str = '.', extension='csv'):
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


def read_csv_to_dataframe(filename: str, col_sep=',', index_col=None) -> pd.DataFrame:
    return pd.read_csv(filepath_or_buffer=filename, sep=col_sep, index_col=index_col, skip_blank_lines=True,
                       quoting=QUOTE_ALL)


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
