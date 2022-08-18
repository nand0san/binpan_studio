from .starters import AesCipher
from .logs import Logs
from .time_helper import convert_milliseconds_to_str
# from .time_helper import convert_utc_ms_column_to_time_zone
from .wallet import get_spot_balances_df, get_spot_balances_total_value, get_spot_trades_list, get_margin_trades_list

from pandas import DataFrame
import requests
from time import sleep

msg_logger = Logs(filename='./logs/msg_logger.log', name='msg_logger', info_level='INFO')

cipher_object = AesCipher()

try:
    from secret import encoded_chat_id, encoded_telegram_bot_id

except Exception as exc:
    msg = """
Not found telegram bot key or chat key for the telegram message module.

Example adding bot key and chat id:
    
    from binpan import handlers

    # from @BotFather, get the bot api key
    handlers.files.add_any_key(key="xxxxxxxxxx", key_name="encoded_telegram_bot_id")

    # write to your bot and then get your chat id from https://api.telegram.org/bot<YourBOTToken>/getUpdates
    handlers.files.add_any_key(key="xxxxxxxxxx", key_name="encoded_chat_id")
"""
    msg_logger.warning(msg)
    encoded_telegram_bot_id = ''
    encoded_chat_id = ''


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


    :param str msg: A message
    :param str parse_mode: Parsing telegram message. Default is 'Markdown&text'.
    :param disable_notification: Avoids getting notified by the telegram message. Only working in group chats.
        https://stackoverflow.com/questions/64360221/disable-notification-in-python-telegram-bot
    :param str alt_enc_bot_id: Optional alternative bot api key.
    :param str alt_chat_id: Optional alternative chat id.
    :return: Telegrams API response.
    """

    if type(msg) == dict:
        bot_message = str(msg.copy())
    else:
        bot_message = str(msg)

    if not alt_enc_bot_id:
        alt_enc_bot_id = cipher_object.decrypt(encoded_telegram_bot_id)
    if not alt_chat_id:
        alt_chat_id = cipher_object.decrypt(encoded_chat_id)

    send_text = f'https://api.telegram.org/bot{alt_enc_bot_id}/sendMessage?chat_id={alt_chat_id}&' \
                f'parse_mode={parse_mode}={bot_message}'

    if disable_notification:
        send_text += "&disable_notification=true"

    response = requests.get(send_text)

    msg_logger.info("TELEGRAM " + " ".join(bot_message.replace('\n', ' ').split()))

    return response.json()


def telegram_parse_dict(msg_data: dict, timezone='UTC'):
    """
    Parses a dict and downcast types.

    :param dict msg_data: Dict to parse as message.
    :param str timezone: A time zone to parse more human readable any field containing "time" in the name.
    :return str: A markdown v1 parsed telegram string.
    """
    msg_dict = msg_data.copy()
    parsed_msg = ""
    for k, v in msg_dict.items():
        try:
            fv1 = float(v)
        except:
            fv1 = v
        try:
            fv2 = int(fv1)
            assert fv2 == fv1
        except:
            fv2 = fv1
        if 'pct' in k:
            fv2 = fv2 * 100
            row = f"*{k}* : `{fv2:.2f}` \n"
        elif type(fv2) == float:
            row = f"*{k}* : `{fv2:.8f}` \n"
        elif type(fv2) == int:
            row = f"*{k}* : `{fv2}` \n"
        else:
            if 'time' in k and k != 'timeInForce':
                date = convert_milliseconds_to_str(ms=fv2, timezoned=timezone)
                row = f"*{k}* : {date} \n"
            else:
                row = f"*{k}* : {fv2} \n"
        parsed_msg += row
    return parsed_msg.replace('_', ' ')


# futuro bot #


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


def telegram_parse_dataframe_markdown(data: DataFrame,
                                      indentation: int = 6,
                                      title: str = "Balances") -> str:
    """
    Parses a Dataframe in telegrams message format.

    :param DataFrame data: A dataframe
    :param int indentation: Spaces to insert each tabulation.
    :param str title: a header or title to the result.
    :return str: String parsed with tabulations.
    """
    ret = f"*{title}*: \n"
    cols = data.columns
    ret += "```\n"
    for idx, row in data.iterrows():
        data_row = row.tolist()
        str_row = f" {idx} " + ''.join([f"{cols[i]}: {d:.8f} " for i, d in enumerate(data_row)])
        ret += str(tab_str(str_row, indentation=indentation)) + "\n"
    return ret + "```"


def get_fills_price(original_order_dict: dict,
                    isBuyer: bool,
                    margin: bool,
                    test_mode: bool) -> float:
    """
    Obtain averaged price from order API response or claims last trade.

    :param dict original_order_dict: API order dict response.
    :param bool isBuyer: Sets if was a buyer order to filter for wanted order.
    :param bool margin: Sets margin order if it is to claim.
    :param bool test_mode: Set if test mode is on.
    :return float: Price averaged.

    Example of not totally done order:

    .. code-block::

        {'symbol': 'ACMBUSD',
        'orderId': 53255492,
        'orderListId': -1,
        'clientOrderId': 'DUHlcCqIX2OO298XoAbvEB',
        'transactTime': 1660332903950,
        'price': 4.158,
        'origQty': 21.1,
        'executedQty': 0.0,
        'cummulativeQuoteQty': 0.0,
        'status': 'NEW',
        'timeInForce': 'GTC',
        'type': 'LIMIT',
        'side': 'BUY',
        'fills': []}


    Example of full SPOT response:

    .. code-block::

        {
          "symbol": "BTCUSDT",
          "orderId": 28,
          "orderListId": -1, //Unless OCO, value will be -1
          "clientOrderId": "6gCrw2kRUAF9CvJDGP16IP",
          "transactTime": 1507725176595,
          "price": "0.00000000",
          "origQty": "10.00000000",
          "executedQty": "10.00000000",
          "cummulativeQuoteQty": "10.00000000",
          "status": "FILLED",
          "timeInForce": "GTC",
          "type": "MARKET",
          "side": "SELL",
          "strategyId": 1,               // This is only visible if the field was populated on order placement.
          "strategyType": 1000000        // This is only visible if the field was populated on order placement.
          "fills": [
            {
              "price": "4000.00000000",
              "qty": "1.00000000",
              "commission": "4.00000000",
              "commissionAsset": "USDT",
              "tradeId": 56
            },
            {
              "price": "3999.00000000",
              "qty": "5.00000000",
              "commission": "19.99500000",
              "commissionAsset": "USDT",
              "tradeId": 57
            },
            {
              "price": "3998.00000000",
              "qty": "2.00000000",
              "commission": "7.99600000",
              "commissionAsset": "USDT",
              "tradeId": 58
            },
            {
              "price": "3997.00000000",
              "qty": "1.00000000",
              "commission": "3.99700000",
              "commissionAsset": "USDT",
              "tradeId": 59
            },
            {
              "price": "3995.00000000",
              "qty": "1.00000000",
              "commission": "3.99500000",
              "commissionAsset": "USDT",
              "tradeId": 60
            }
          ]
        }

    Example of FULL MARGIN RESPONSE:

    .. code-block::

        {
          "symbol": "BTCUSDT",
          "orderId": 28,
          "clientOrderId": "6gCrw2kRUAF9CvJDGP16IP",
          "transactTime": 1507725176595,
          "price": "1.00000000",
          "origQty": "10.00000000",
          "executedQty": "10.00000000",
          "cummulativeQuoteQty": "10.00000000",
          "status": "FILLED",
          "timeInForce": "GTC",
          "type": "MARKET",
          "side": "SELL",
          "marginBuyBorrowAmount": 5,       // will not return if no margin trade happens
          "marginBuyBorrowAsset": "BTC",    // will not return if no margin trade happens
          "isIsolated": true,       // if isolated margin
          "fills": [
            {
              "price": "4000.00000000",
              "qty": "1.00000000",
              "commission": "4.00000000",
              "commissionAsset": "USDT"
            },
            {
              "price": "3999.00000000",
              "qty": "5.00000000",
              "commission": "19.99500000",
              "commissionAsset": "USDT"
            },
            {
              "price": "3998.00000000",
              "qty": "2.00000000",
              "commission": "7.99600000",
              "commissionAsset": "USDT"
            },
            {
              "price": "3997.00000000",
              "qty": "1.00000000",
              "commission": "3.99700000",
              "commissionAsset": "USDT"
            },
            {
              "price": "3995.00000000",
              "qty": "1.00000000",
              "commission": "3.99500000",
              "commissionAsset": "USDT"
            }
          ]
        }

    """
    ordered = original_order_dict.copy()
    if 'fills' in ordered.keys():
        if type(ordered['fills']) == dict:
            return float(ordered['fills']['price'])
        elif type(ordered['fills']) == list:
            n = 0
            m = 0
            for fill in ordered['fills']:
                price = float(fill['price'])
                qty = float(fill['qty'])
                n += price * qty
                m += qty
            if m:
                return n / m
        else:
            raise Exception(f"BinPan Error: Error en parseo de orden ejecutada.\n{ordered}")
    else:
        # if it was a buy market order, it has no price, price = 0.0, then get from API trades
        price_ret = float(ordered['price'])
        if price_ret != 0:
            return float(price_ret)

        if test_mode:
            raise Exception(f"Omega Bot error: Parsing fills in test mode.")

    # any other case
    msg_logger.info(f"Get fills from API trades!... wait 3 seconds for trade to appear")
    sleep(3)
    symbol = ordered['symbol']

    if not margin:
        last_trades = get_spot_trades_list(symbol=symbol)
    else:
        last_trades = get_margin_trades_list(symbol=symbol)
    last_trade = [i for i in last_trades if i['isBuyer'] == isBuyer][-1]
    return float(last_trade['price'])


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

    if 'fills' in order_.keys():
        order_['fills'] = get_fills_price(original_order_dict=order_,
                                          isBuyer=isBuyer,
                                          margin=margin,
                                          test_mode=test_mode)
        order_['price'] = order_['fills']
    try:
        order_.pop('clientOrderId', None)
    except Exception as exc:
        msg_logger.debug(f"Error popping key clientOrderId {exc}")
    try:
        order_.pop('orderListId', None)
    except Exception as exc:
        msg_logger.debug(f"Error popping key orderListId {exc}")

    order_['symbol'] = f"*{order_['symbol']}* \n------------------------------------------------"

    if 'time' in order_.keys():
        date = convert_milliseconds_to_str(int(order_['time']), timezoned=timezoned)
        order_.update({'time': date})

    if 'updateTime' in order_.keys():
        date = convert_milliseconds_to_str(int(order_['updateTime']), timezoned=timezoned)
        order_.update({'updateTime': date})

    if 'quantity' in order_.keys():
        order_.update({'quantity': f"`{order_['quantity']}`"})

    if 'price' in order_.keys():
        order_.update({'price': f"`{order_['price']}`"})

    if 'origQty' in order_.keys():
        order_.update({'origQty': f"`{order_['origQty']}`"})

    if 'executedQty' in order_.keys():
        order_.update({'executedQty': f"`{order_['executedQty']}`"})

    if 'stopPrice' in order_.keys():
        order_.update({'stopPrice': f"`{order_['stopPrice']}`"})

    if 'fills' in order_.keys():
        order_.update({'fills': f"`{order_['fills']}`"})

    for key, val in order_.items():
        if val:
            line = rf"*{key}*: {val}" + "\n"
            parsed += line

    return parsed


def send_balances(convert_to: str = 'BUSD') -> float:
    """
    Sends telegram message with total value of spot wallet in selected coin.

    It returns free and locked assets value added.

    :param str convert_to: A Binance coin.
    :return float: total value of wallet.

    """
    balances_df = get_spot_balances_df()
    parsed_balances = telegram_parse_dataframe_markdown(balances_df)

    # add total value in usdt
    total_value = get_spot_balances_total_value(balances_df=balances_df,
                                                convert_to=convert_to)

    parsed_balances += f"\n*Total value*: `{total_value}` \n*Quote*: `{convert_to}`"
    telegram_bot_send_text(parsed_balances)
    return total_value
