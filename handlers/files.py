from csv import QUOTE_ALL
from time import time
from sys import path
from os import listdir, path, mkdir, replace, makedirs
import importlib
import pandas as pd

from .starters import AesCipher, import_secret_module
from .logs import Logs
from .exceptions import BinPanException, MissingBinanceApiData

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


def save_dataframe_to_csv(filename: str, data: pd.DataFrame, col_sep=',', index=False, timestamp=True) -> None:
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


def find_csvs_in_path(files_path: str = '.', extension='csv', name_filter: str = None):
    """
    Locate all files with desired extension.

    :param str files_path: A path.
    :param str extension: A extension like "csv".
    :param str name_filter: A filter to apply to files. Example: "klines" or "aggTrades" or "atomicTrades".
    :return:
    """
    ret = []
    for r in listdir(files_path):
        full_path = path.join(files_path, r)
        if path.isfile(full_path):
            if name_filter:
                if name_filter in full_path:
                    ret.append(full_path)
            else:
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


def extract_filename_metadata(filename: str, expected_data_type: str, expected_symbol: str = None, expected_interval: str = None,
                              expected_timezone: str = None) -> tuple:
    """
    Extract metadata from a filename. Symbol, tick interval, timezone and start timestamp and end timestamp. Also data type like atomic
    trades or klines.

    :param filename: A filename with expected format. Example: LTCBTC 1m Europe-Madrid optional comment klines 1691332020000
     1691391959999.csv
    :param str expected_data_type: Expected data type. Example: "klines" or "aggTrades" or "atomicTrades"
    :param str expected_interval: Expected tick interval. Example: "1m" or "1h".
    :param str expected_symbol: Expected symbol. Example: "LTCBTC"
    :param str expected_timezone: Expected timezone. Example: "Europe/Madrid"
    :return: A dict with metadata.
    """
    filename = str(path.basename(filename))

    try:
        assert expected_data_type in ['klines', 'aggTrades',
                                      'atomicTrades'], (f'Expected data type is not valid. Expected "klines" or "aggTrades" or '
                                                        f'"atomicTrades". Received {expected_data_type}')
    except AssertionError as e:
        files_logger.error(f"BinPan Exception: {e}")
        raise e

    symbol = filename.split(' ')[0]

    if expected_data_type == 'klines':
        tick_interval = filename.split(' ')[1]
        time_zone = filename.split(' ')[2].replace("-", "/")
    else:
        tick_interval = None
        time_zone = filename.split(' ')[1].replace("-", "/")

    # comments can go before timezone
    data_type = filename.split(' ')[-3]
    start_timestamp = filename.split(' ')[-2]
    end_timestamp = filename.split(' ')[-1].replace('.csv', '')

    if expected_symbol:
        try:
            assert symbol == expected_symbol, f'Expected symbol is not valid. Expected {expected_symbol}. Received {symbol}'
        except AssertionError as e:
            files_logger.error(f"BinPan Exception: {e}")
            raise e
    if expected_interval:
        try:
            assert tick_interval == expected_interval, (f'Expected tick interval is not valid. Expected '
                                                        f'{expected_interval}. Received {tick_interval}')
        except AssertionError as e:
            files_logger.error(f"BinPan Exception: {e}")
            raise e
    if expected_timezone:
        try:
            assert time_zone == expected_timezone, f'Expected timezone is not valid. Expected {expected_timezone}. Received {time_zone}'
        except AssertionError as e:
            files_logger.error(f"BinPan Exception: {e}")
            raise e
    return symbol.upper(), tick_interval, time_zone, data_type, start_timestamp, end_timestamp


def read_csv_to_dataframe(filename: str, col_sep: str = ',', index_col: str = None, index_time_zone: str = None,
                          symbol: str = None, secondary_index_col: str = None) -> pd.DataFrame:
    """
    Creates a csv file from a dataframe.

    :param str filename: The file name with or without path.
    :param str col_sep: Column separator. Default is ","
    :param bool index_col: Name of the column to use as index. Default is None.
    :param str secondary_index_col: In case of duplicated index, it will be used as secondary criterio for sorting data.
    :param str index_time_zone: Time zone to use in index. Default is None.
    :param str symbol: Symbol to use in index name. Default is None.
    :return pd.DataFrame: A dataframe with data in columns using file rows header.
    """
    df_ = pd.read_csv(filepath_or_buffer=filename, sep=col_sep, skip_blank_lines=True, quoting=QUOTE_ALL)
    if index_col:
        # verifica si hay duplicados en el index_col
        if df_.index.duplicated().any():
            files_logger.warning(f"BinPan Warning: Duplicated index in {filename}")
            # apply sort criterion with second index
            if secondary_index_col:
                df_.sort_values(by=[index_col, secondary_index_col], inplace=True)
            else:
                raise BinPanException(f"BinPan Exception: Duplicated index in {filename} and no secondary index provided")

        df_.set_index(index_col, inplace=True, drop=False)

        if index_time_zone:
            df_.index = pd.to_datetime(df_.index, unit='ms')
            if not df_.index.tz:
                df_.index = df_.index.tz_localize(tz='UTC')
                df_.index = df_.index.tz_convert(index_time_zone)
            else:
                df_.index = df_.index.tz_convert(index_time_zone)
            if symbol:
                df_.index.name = f"{symbol} {index_time_zone}"
    return df_


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


def select_file(path='.', extension='csv', symbol: str = None, tick_interval: str = None, name_filter: str = None) -> str:
    """
    Selects from files in the path with the extension passed.

    :param str path: Path to search files.
    :param str extension: Extension of interesting files to select.
    :param str symbol: Symbol to filter files just to alert if file has other symbol.
    :param str tick_interval: Tick interval to filter files just to alert if file has other tick interval.
    :param str name_filter: Filter to apply to files. Example: "klines" or "aggTrades" or "atomicTrades".
    :return str: a filename.
    """
    print("File selection menu:")
    files = find_csvs_in_path(files_path=path, extension=extension, name_filter=name_filter)
    files = [i for i in files if i.lower().endswith(extension.lower())]
    for i, file in enumerate(files):
        print(f"{i}: {file}")
    selection = input("Insert file menu number: ")
    if symbol:
        if symbol not in files[int(selection)]:
            files_logger.warning(f"BinPan Warning: Selected file has not the expected symbol {symbol}")
    if tick_interval:
        if " " + tick_interval + " " not in files[int(selection)]:
            files_logger.warning(f"BinPan Warning: Selected file has not the expected tick interval {tick_interval}")
    return files[int(selection)]


###################
# API AND SECRETS #
###################
def get_encoded_secrets() -> tuple:
    try:
        secret_module = import_secret_module()
        importlib.reload(secret_module)
        api_key = secret_module.api_key
        api_secret = secret_module.api_secret
    except Exception:
        print(f"Binance Api key or Api Secret are needed for this API calls.")
        fill_secrets_file()
        try:
            secret_module = import_secret_module()
            importlib.reload(secret_module)
            api_key = secret_module.api_key
            api_secret = secret_module.api_secret
        except Exception:
            raise MissingBinanceApiData(f"Binance Api key or Api Secret not found.")
    return api_key, api_secret


def get_encoded_telegram_secrets() -> tuple:
    try:
        secret_module = import_secret_module()
        importlib.reload(secret_module)
        encoded_telegram_bot_id = secret_module.encoded_telegram_bot_id
        encoded_chat_id = secret_module.encoded_chat_id
    except Exception:
        print(f"Telegram bot_id and chat id not found.")
        fill_telegram_secrets_file()
        try:
            secret_module = import_secret_module()
            importlib.reload(secret_module)
            encoded_telegram_bot_id = secret_module.encoded_telegram_bot_id
            encoded_chat_id = secret_module.encoded_chat_id
        except Exception:
            # raise MissingTelegramApiData(f"Telegram bot_id and chat id not found.")
            files_logger.warning(f"BinPan Warning: Telegram bot_id and chat id not found.")
            encoded_telegram_bot_id, encoded_chat_id = "", ""
    return encoded_telegram_bot_id, encoded_chat_id


def get_encoded_database_secrets() -> str:
    try:
        secret_module = import_secret_module()
        importlib.reload(secret_module)
        postgresql_password = secret_module.postgresql_password
    except Exception:
        print(f"postgresql_password not found in secret.py.")
        fill_database_secrets_file()
        try:
            secret_module = import_secret_module()
            importlib.reload(secret_module)
            postgresql_password = secret_module.postgresql_password
        except Exception:
            files_logger.warning(f"BinPan Warning: PostgreSQL password not found.")
            postgresql_password = ""
    return postgresql_password


def fill_secrets_file() -> None:
    """
    Creates a file called secret.py or adds to it with the API key and API secret encrypted values.

    :return: None
    """
    api_key = input(f"Please, enter your Binance Api Key: ")
    add_api_key(api_key_value=api_key)
    print("Key encrypted and saved.")
    api_secret = input(f"Please, enter your Binance Api Secret: ")
    add_api_secret(api_secret_value=api_secret)
    print("Secret encrypted and saved.")


def fill_telegram_secrets_file() -> None:
    """
    Creates a file called secret.py or adds to it with the telegram bot_id and chat_id encrypted values.

    :return: None
    """
    bot = input(f"Please, enter your Telegram bot token for send messages or leave it empty: ")
    add_any_key(key=bot, key_name="encoded_telegram_bot_id")
    print("Telegram bot token encrypted and saved.")
    chat = input(f"Please, enter your Telegram chat id for send messages to you or leave it empty: ")
    add_any_key(key=chat, key_name="encoded_chat_id")
    print("Chat encrypted and saved.")


def fill_database_secrets_file() -> None:
    """
    Creates a file called secret.py or adds to it with the postgresql password encrypted values.

    :return: None
    """
    password = input(f"Please, enter your postgresql password: ")
    add_any_key(key=password, key_name="postgresql_password")
    print("postgresql password encrypted and saved.")


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
            lines.append(line + "\n")
    encryptor = cipher_object.encrypt(api_key_value)
    lines.append(f'api_key = "{encryptor}"\n')
    save_file(filename=filename, data=lines)


def add_api_secret(api_secret_value: str) -> None:
    """
    Checks if exists in secret.py file and if not, then adds a line with the API secret encrypted value for working with the package.

    :param str api_secret_value: API key
    """
    filename = "secret.py"
    saved_data = read_file(filename=filename)
    lines = [line + '\n' for line in saved_data if (not line.startswith('api_secret')) and line]
    encryptor = cipher_object.encrypt(api_secret_value)
    lines.append(f'api_secret = "{encryptor}"\n')
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
                lines.append(line + '\n')
    encryptor = cipher_object.encrypt(key)
    lines.append(f'{key_name} = "{encryptor}"\n')
    save_file(filename=filename, data=lines)
