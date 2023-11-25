"""
Telegram and messages control.
"""
import requests
import pandas as pd
import numpy as np
import datetime

from .starters import AesCipher
from .files import get_encoded_telegram_secrets
from .logs import Logs
from .time_helper import convert_milliseconds_to_str
from .wallet import get_spot_balances_df, get_spot_balances_total_value

msg_logger = Logs(filename='./logs/msg_logger.log', name='msg_logger', info_level='INFO')

cipher_object = AesCipher()


def telegram_bot_send_text(msg: dict or str,
                           parse_mode='Markdown&text',
                           disable_notification=False,
                           alt_enc_bot_id: str = None,
                           alt_chat_id: str = None) -> dict:
    """
    Sends a telegram message.

    It takes bot key and chat id from secret encrypted file. It is required to create previously the bot and chat keys with the
    variable names *encoded_chat_id*, *encoded_telegram_bot_id*.

    Example adding bot key and chat id:

    .. code-block:: python

        from binpan import handlers

        # from @BotFather, get the bot api key
        handlers.files.add_any_key(key="xxxxxxxxxx", key_name="encoded_telegram_bot_id")

        # write to your bot and then get your chat id from https://api.telegram.org/bot<YourBOTToken>/getUpdates
        handlers.files.add_any_key(key="xxxxxxxxxx", key_name="encoded_chat_id")


    :param str or dict msg: A message
    :param str parse_mode: Parsing telegram message. Default is 'Markdown&text'.
    :param disable_notification: Avoids getting notified by the telegram message. Only working in group chats.
        https://stackoverflow.com/questions/64360221/disable-notification-in-python-telegram-bot
    :param str alt_enc_bot_id: Optional alternative bot api key.
    :param str alt_chat_id: Optional alternative chat id.
    :return: Telegrams API response.
    """
    encoded_telegram_bot_id, encoded_chat_id = get_encoded_telegram_secrets()

    if type(msg) is dict:
        bot_message = str(msg.copy())
    else:
        bot_message = str(msg)

    # telegram keys
    if not alt_enc_bot_id:
        alt_enc_bot_id = encoded_telegram_bot_id
    if not alt_chat_id:
        alt_chat_id = encoded_chat_id
    alt_enc_bot_id = cipher_object.decrypt(alt_enc_bot_id)
    alt_chat_id = cipher_object.decrypt(alt_chat_id)

    send_text = f'https://api.telegram.org/bot{alt_enc_bot_id}/sendMessage?chat_id={alt_chat_id}&' \
                f'parse_mode={parse_mode}={bot_message}'
    if disable_notification:
        send_text += "&disable_notification=true"
    response = requests.get(send_text)
    msg_logger.info("TELEGRAM " + " ".join(bot_message.replace('\n', ' ').split()))
    return response.json()


def sort_mixed_dict(d: dict) -> dict:
    """
    Sort a dictionary that may contain mixed types (int, float, str, etc.)

    :param dict d: The dictionary to sort.
    :return dict: A new sorted dictionary.
    """
    int_dict = {k: v for k, v in d.items() if isinstance(v, int)}
    float_dict = {k: v for k, v in d.items() if isinstance(v, float)}
    str_dict = {k: v for k, v in d.items() if isinstance(v, str)}
    dict_dict = {k: v for k, v in d.items() if isinstance(v, dict)}

    int_dict = {k: v for k, v in sorted(int_dict.items(), key=lambda item: item[1], reverse=True)}
    float_dict = {k: v for k, v in sorted(float_dict.items(), key=lambda item: item[1], reverse=True)}

    sorted_dict = {**int_dict, **float_dict, **str_dict, **dict_dict}
    return sorted_dict


def telegram_parse_dict(msg_data: dict, order: str = None, timezone="Europe/Madrid"):
    """
    Parses a dictionary to a telegram message.

    :param msg_data: A dictionary with the data to be parsed.
    :param order: Ascending or descending order. Example: "asc" or "desc".
    :param timezone: A string with the timezone.
    :return: A string with the parsed message.
    """

    def downcast_value(value):
        try:
            assert int(value) == float(value)
            return int(value)
        except:
            try:
                return float(value)
            except (ValueError, TypeError):
                return value

    formatted_dict = {}
    for k, v in msg_data.items():
        if isinstance(v, tuple):
            value = tuple(downcast_value(i) for i in v)
        elif isinstance(v, dict):
            value = telegram_parse_dict(v)
        elif isinstance(v, list):
            value = [downcast_value(i) for i in v]
        else:
            value = downcast_value(v)
        formatted_dict[k] = value

    if order:
        message_tuples = []
        for k, v in formatted_dict.items():
            if isinstance(v, tuple):
                value = -np.inf
                if type(v[0]) is int or type(v[0]) is float:
                    value = v[0]
            else:
                value = v
            message_tuples.append((k, value))
        if order == "asc":
            message_tuples.sort(key=lambda x: x[1])
        elif order == "desc":
            message_tuples.sort(key=lambda x: x[1], reverse=True)
    else:
        message_tuples = formatted_dict.items()

    parsed_msg = ""

    for k, _ in message_tuples:
        fv = formatted_dict[k]
        if isinstance(fv, tuple):
            tuple_str = " ; ".join([f"`{i}`" for i in fv])
            row = f"*{k}*: {tuple_str} \n"
        elif 'pct' in k and not "kline" in k and not "time" in k and not "timestamp" in k and not "trade" in k and not "Trade" in k:
            if isinstance(fv, float):
                fv *= 100
                row = f"*{k}* : `{fv:.2f}` \n"
            else:
                row = f"*{k}* : `{fv}` \n"
        elif "_delay" in k and fv:
            fv /= 1000
            row = f"*{k}* : `{fv:.2f}` seconds\n"

        elif 'time' in k or 'timestamp' in k:
            if isinstance(fv, datetime.datetime):
                fv = fv.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(fv, int):
                fv = convert_milliseconds_to_str(fv, timezoned=timezone)
            else:
                fv = fv
            row = f"*{k}* : `{fv}` \n"
        elif isinstance(fv, float) or isinstance(fv, np.float64) or isinstance(fv, np.float32) or isinstance(fv, np.float16):
            row = f"*{k}* : `{fv:.8f}` \n"
        elif isinstance(fv, int) or isinstance(fv, np.int64) or isinstance(fv, np.int32) or isinstance(fv, np.int16):
            row = f"*{k}* : `{fv}` \n"
        elif isinstance(fv, dict):
            row = f"*{k}* : \n{telegram_parse_dict(msg_data=fv, timezone=timezone)} \n"
        else:
            row = f"*{k}*: `{fv}` \n"

        parsed_msg += row

    return parsed_msg.replace('_', ' ').replace("Decimal('", "`").replace("')", "`").replace("{", "").replace("}", "")


def tab_str(text: str, indentation=8) -> str:
    """
    Indentation for telegram messages.
    :param str text: A message with spaces to insert tabulations in.
    :param int indentation: The spaces quantity to insert.
    :return str: Tabulated message.
    """
    sp = text.split()
    ret = ""
    for i in sp:
        length = len(i)
        ind = (indentation - length) % indentation
        ret += i + " " * ind
    return ret


def telegram_parse_dataframe_markdown(data: pd.DataFrame, indentation: int = 10, title: str = "Balances") -> str:
    """
    Parses a Dataframe in telegrams message format.

    :param DataFrame data: A dataframe
    :param int indentation: Spaces to insert each tabulation.
    :param str title: a header or title to the result.
    :return str: String parsed with tabulations.
    """
    ret = f"*{title}* *{list(data.columns)}*: \n"
    cols = data.columns
    ret += "```\n"
    for idx, row in data.iterrows():
        data_row = row.tolist()
        str_row = f" {idx} " + ''.join([f"{cols[i]}: {d} " for i, d in enumerate(data_row)])
        ret += str(tab_str(str_row, indentation=indentation)) + "\n"
    return ret + "```"


def telegram_parse_order_markdown(original_order: dict,
                                  isBuyer: bool,
                                  margin: bool,
                                  test_mode: bool,
                                  timezoned='Europe/Madrid'
                                  ):
    """
    Parses API order response to telegram string.

    :param dict original_order: API order response.
    :param bool isBuyer: Sets if was a buyer order to filter for wanted order.
    :param bool margin: Sets margin order if it is to claim.
    :param bool test_mode: Set if test mode is on.
    :param str timezoned: a time zone like 'Europe/Madrid' for human readable timestamps.
    :return str: Parsed telegram style  string.
    """
    order_ = original_order.copy()
    parsed = ""

    if 'orderReports' in order_.keys():
        p_stop = telegram_parse_order_markdown(order_['orderReports'][0], isBuyer=isBuyer, margin=margin, test_mode=test_mode)
        p_limit = telegram_parse_order_markdown(order_['orderReports'][1], isBuyer=isBuyer, margin=margin, test_mode=test_mode)
        stop_parsed = f"\n{p_stop}\n"
        limit_parsed = f"\n{p_limit}\n"
    else:
        stop_parsed = 'Possible error with stop_parsed and orderReports'
        limit_parsed = 'Possible error with and limit_parsed orderReports'

    if 'time' in order_.keys():
        date = convert_milliseconds_to_str(int(order_['time']), timezoned=timezoned)
        order_.update({'time': date})

    if 'transactTime' in order_.keys():
        date = convert_milliseconds_to_str(int(order_['transactTime']), timezoned=timezoned)
        order_.update({'transactTime': date})

    if 'updateTime' in order_.keys():
        date = convert_milliseconds_to_str(int(order_['updateTime']), timezoned=timezoned)
        order_.update({'updateTime': date})

    if 'quantity' in order_.keys():
        order_.update({'quantity': f"{order_['quantity']}"})

    if 'price' in order_.keys():
        order_.update({'price': f"{order_['price']}"})

    if 'origQty' in order_.keys():
        order_.update({'origQty': f"{order_['origQty']}"})

    if 'executedQty' in order_.keys():
        order_.update({'executedQty': f"{order_['executedQty']}"})

    if 'stopPrice' in order_.keys():
        order_.update({'stopPrice': f"{order_['stopPrice']}"})

    if 'fills' in order_.keys():
        order_.update({'fills': f"{order_['fills']}"})

    for key, val in order_.items():
        if key == 'orderReports':
            line = stop_parsed
            parsed += line
            line = limit_parsed
            parsed += line
        elif val:
            line = rf"*{key}* : `{val}` " + "\n"
            parsed += line

    return parsed


def send_balances(decimal_mode: bool,
                  api_key: str,
                  api_secret: str,
                  convert_to: str = 'BUSD') -> float:
    """
    Sends telegram message with total value of spot wallet in selected coin.

    It returns free and locked assets value added.
    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param str convert_to: A Binance coin.
    :return float: total value of wallet.

    """
    balances_df = get_spot_balances_df(decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
    parsed_balances = telegram_parse_dataframe_markdown(balances_df)

    # add total value in usdt
    total_value = get_spot_balances_total_value(balances_df=balances_df,
                                                convert_to=convert_to,
                                                decimal_mode=decimal_mode,
                                                api_key=api_key,
                                                api_secret=api_secret)

    parsed_balances += f"\n*Total value*: `{total_value}` \n*Quote*: `{convert_to}`"
    telegram_bot_send_text(parsed_balances)
    return total_value
