import pandas as pd
from decimal import Decimal as dd
from time import sleep

from .logs import Logs
from .quest import api_raw_signed_get, api_raw_signed_post
from .time_helper import convert_milliseconds_to_str
from .time_helper import convert_string_to_milliseconds
from .market import get_prices_dic, convert_coin

wallet_logger = Logs(filename='./logs/wallet_logger.log', name='wallet_logger', info_level='INFO')


##########
# Helper #
##########

def convert_str_date_to_ms(date: str or int,
                           time_zone: str):
    """
    Converts dates strings formatted as "2022-05-11 06:45:42" to timestamp in milliseconds.

    :param str or int date: Date to check format.
    :param time_zone: A time zone like 'Europe/Madrid'
    :return int: Milliseconds of timestamp.
    """
    if type(date) == str:
        date = convert_string_to_milliseconds(date, timezoned=time_zone)
    return date


##########
# WALLET #
##########


def daily_account_snapshot(account_type: str,
                           decimal_mode: bool,
                           api_key: str,
                           api_secret: str,
                           startTime: int or str = None,
                           endTime: int = None,
                           limit=30,
                           time_zone=None) -> pd.DataFrame:
    """
    The query time period must be inside the previous 30 days.
    Support query within the last month, one month only.
    If startTime and endTime not sent, return records of the last days by default.

    Weight(IP): 2400

    :param str account_type: SPOT or MARGIN (cross)
    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param int limit: Days limit. Default 30.
    :param int or str startTime: Can be integer timestamp in milliseconds or formatted string: 2022-05-11 06:45:42
    :param int or str endTime: Can be integer timestamp in milliseconds or formatted string: 2022-05-11 06:45:42
    :param str time_zone: A time zone to parse index like 'Europe/Madrid'
    :return pd.DataFrame: Pandas DataFrame

    """
    startTime = convert_str_date_to_ms(date=startTime, time_zone=time_zone)
    endTime = convert_str_date_to_ms(date=endTime, time_zone=time_zone)

    available = ['SPOT', 'MARGIN']
    if account_type not in available:
        exc = f"BinPan error: {account_type} not in {available}"
        wallet_logger.error(exc)
        raise Exception(exc)
    elif limit > 30 or limit < 7:
        exc = f"BinPan error: {limit} is over the maximum 30 or under the minimum 7."
        wallet_logger.error(exc)
        raise Exception(exc)

    ret = api_raw_signed_get(endpoint='/sapi/v1/accountSnapshot',
                             params={'type': account_type,
                                     'startTime': startTime,
                                     'endTime': endTime,
                                     'limit': limit},
                             weight=2400, decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
    rows = []
    for update in ret['snapshotVos']:
        updateTime = update['updateTime']
        data = update['data']
        totalAssetOfBtc = float(data['totalAssetOfBtc'])
        if account_type == 'MARGIN':
            assets_field = 'userAssets'
        else:
            assets_field = 'balances'
        for balance in data[assets_field]:
            free = float(balance['free'])
            locked = float(balance['locked'])
            if free > 0 or locked > 0:
                row = {'updateTime': updateTime,
                       'totalAssetOfBtc': totalAssetOfBtc,
                       'asset': balance['asset'],
                       'free': free,
                       'locked': locked}
                if time_zone:
                    row['datetime'] = convert_milliseconds_to_str(ms=updateTime, timezoned=time_zone)
                rows.append(row)

    if time_zone:
        ret = pd.DataFrame(rows)
        if not ret.empty:
            ret = ret.set_index('datetime').sort_index()
            ret.index.name = f"{account_type} {time_zone}"
    else:
        ret = pd.DataFrame(rows)
        if not ret.empty:
            ret = ret.set_index('updateTime').sort_index()
            ret.index.name = f"{account_type} timestamp"
    return ret


def assets_convertible_dust(decimal_mode: bool,
                            api_key: str,
                            api_secret: str) -> dict:
    """
    Assets dust that can be converted to BNB.
    Weight(IP): 1

    :param bool decimal_mode: It flags to work in decimal mode.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :return dict: A dictionary.

    .. code-block::

       {
            "details": [
                {
                    "asset": "ADA",
                    "assetFullName": "ADA",
                    "amountFree": "6.21",   //Convertible amount
                    "toBTC": "0.00016848",  //BTC amount
                    "toBNB": "0.01777302",  //BNB amount（Not deducted commission fee）
                    "toBNBOffExchange": "0.01741756", //BNB amount（Deducted commission fee）
                    "exchange": "0.00035546" //Commission fee
                }
            ],
            "totalTransferBtc": "0.00016848",
            "totalTransferBNB": "0.01777302",
            "dribbletPercentage": "0.02"     //Commission fee
        }

    """
    yn = input(f"This command may change or convert assets from your wallet!!! are you sure? (y/n)")
    if yn.upper().startswith('Y'):
        return api_raw_signed_post(endpoint='/sapi/v1/asset/dust-btc',
                                   weight=1,
                                   decimal_mode=decimal_mode,
                                   api_key=api_key,
                                   api_secret=api_secret)
    else:
        wallet_logger.warning("Canceled!")


##########
# trades #
##########


def get_fills_price(original_order_dict: dict,
                    isBuyer: bool,
                    margin: bool,
                    test_mode: bool,
                    operation_time: int,
                    decimal_mode: bool,
                    api_key: str,
                    api_secret: str) -> float:
    """
    Obtain averaged price from order API response or claims last trade.

    :param dict original_order_dict: API order dict response.
    :param bool isBuyer: Sets if was a buyer order to filter for wanted order.
    :param bool margin: Sets margin order if it is to claim.
    :param bool test_mode: Set if test mode is on.
    :param int operation_time: Time limit of operation to analyze.
    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
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
          "clientOrderId": "6gCrw2kRUFF9CvJDWP16IP",
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
          "clientOrderId": "6gCrw2kRUFF9CvJDWP16IP",
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

    STOP LIMIT EXAMPLE requested order response:

    .. code-block::

         {'symbol': 'CHZBUSD',
          'orderId': 212320642,
          'orderListId': 72689217,
          'clientOrderId': 'FLc7AHuuZCpChiH99eHtZh',
          'price': 0.2139,
          'origQty': 224.0,
          'executedQty': 224.0,
          'cummulativeQuoteQty': 47.9584,
          'status': 'FILLED',
          'timeInForce': 'GTC',
          'type': 'STOP_LOSS_LIMIT',
          'side': 'SELL',
          'stopPrice': 0.2141,
          'icebergQty': '0.00000000',
          'time': 1661203865507,
          'updateTime': 1661203966780,
          'isWorking': True,
          'origQuoteOrderQty': 0.0},

    """
    ordered = original_order_dict.copy()

    if decimal_mode:
        my_type = dd
    else:
        my_type = float

    if test_mode and 'STOP_LOSS' in ordered['type']:
        return my_type(ordered['stopPrice'])

    if test_mode and 'price' in ordered.keys() and ordered['price']:
        # MUST RETURN FOR TEST ORDERS
        return my_type(ordered['price'])

    if 'fills' in ordered.keys():
        if type(ordered['fills']) == dict:
            if ordered['fills']['price']:
                return my_type(ordered['fills']['price'])

        elif type(ordered['fills']) == list:
            n = 0
            m = 0
            for fill in ordered['fills']:
                price = my_type(fill['price'])
                qty = my_type(fill['qty'])
                n += price * qty
                m += qty
            if m:
                return n / m
        else:
            raise Exception(f"BinPan Error: Error en parseo de orden ejecutada.\n{ordered}")

    # any other case wait 3 times for order to appear
    fills_price = None
    tour = 0
    symbol = ordered['symbol']

    while not fills_price and tour < 5:
        wallet_logger.info(f"Get fills from API trades!... wait 5 seconds for trade to appear. Round:{tour}")

        if not margin:
            last_trades = get_spot_trades_list(symbol=symbol, decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
        else:
            last_trades = get_margin_trades_list(symbol=symbol, decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)

        try:
            # trades return time field with a timestamp
            fills_price = [i for i in last_trades if i['isBuyer'] == isBuyer and int(i['time']) >= operation_time][-1]['price']
            tour += 1
        except IndexError:
            sleep(5)

    return my_type(fills_price)


def get_fills_qty(original_order_dict: dict,
                  api_key: str,
                  api_secret: str,
                  margin: bool,
                  isBuyer: bool,
                  operation_time: int,
                  decimal_mode: bool) -> float or dd:
    """
    Extracts order quantity from an order response or checks a trade from API.

    :param dict original_order_dict: A putting order response from API dict.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param bool margin: Checks if margin or not.
    :param bool isBuyer: Sets if buy or sell side.
    :param int operation_time: A limit old timestamp to verify existence of order.
    :param bool decimal_mode: Sets Decimal operating mode.
    :return float or decimal.Decimal: Returns the needed value.
    """
    ordered = original_order_dict.copy()
    wallet_logger.debug(f"fills_qty: {ordered}")

    ret = None
    if decimal_mode:
        my_type = dd
    else:
        my_type = float

    if 'fills' in ordered.keys():

        if type(ordered['fills']) == dict:
            return my_type(ordered['fills']['qty'])

        elif type(ordered['fills']) == list:
            m = 0
            for fill in ordered['fills']:
                qty = my_type(fill['qty'])
                m += qty
            return m
        else:
            raise Exception(f"BinPan Error: Error parsing order.\n{ordered}")

    else:
        # si fue una orden de compra a precio de mercado no tiene precio, price = 0.0, cantidad ?????
        if 'test_quantity' in ordered.keys():
            ret = ordered['test_quantity']
        elif 'executedQty' in ordered.keys():
            qty_ret = my_type(ordered['executedQty'])
            if qty_ret != 0:
                return qty_ret
        else:
            fills_qty = None
            tour = 0
            symbol = ordered['symbol']

            while not fills_qty and tour < 5:

                wallet_logger.info(f"Get fills qty from API trades!... wait 5 seconds for trade to appear. Round:{tour}")

                if not margin:
                    last_trades = get_spot_trades_list(symbol=symbol, decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
                else:
                    last_trades = get_margin_trades_list(symbol=symbol, decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)

                try:
                    # trades return time field with a timestamp
                    fills_qty = [i for i in last_trades if i['isBuyer'] == isBuyer and int(i['time']) >= operation_time][-1]['qty']
                    tour += 1
                except IndexError:
                    sleep(5)
            ret = fills_qty

    return my_type(ret)


def get_spot_trades_list(symbol: str,
                         decimal_mode: bool,
                         api_key: str,
                         api_secret: str,
                         limit: int = 1000,
                         orderId: int = None,
                         startTime: int = None,
                         endTime: int = None,
                         fromId: int = None,
                         recvWindow: int = 10000) -> list:
    """
    Get trades for a specific account and symbol.

    Weight(IP): 10

    :param str symbol: Symbol's trades.
    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param int limit: Default 500; max 1000.
    :param fromId: TradeId to fetch from. Default gets most recent trades. If fromId is set, it will get id >= that fromId. Otherwise,
       most recent trades are returned.
    :param int endTime: Optional.
    :param int startTime: Optional.
    :param int orderId: This can only be used in combination with symbol.
    :param int recvWindow: The value cannot be greater than 60000
    :return list: A list.

    Example:

           .. code-block::

                     [
                        {
                            "symbol": "BNBBTC",
                            "id": 28457,
                            "orderId": 100234,
                            "orderListId": -1, //Unless OCO, the value will always be -1
                            "price": "4.00000100",
                            "qty": "12.00000000",
                            "quoteQty": "48.000012",
                            "commission": "10.10000000",
                            "commissionAsset": "BNB",
                            "time": 1499865549590,
                            "isBuyer": true,
                            "isMaker": false,
                            "isBestMatch": true
                          }
                       ]
    """
    endpoint = '/api/v3/myTrades'
    return api_raw_signed_get(endpoint=endpoint,
                              params={'limit': limit,
                                      'symbol': symbol,
                                      'orderId': orderId,
                                      'startTime': startTime,
                                      'endTime': endTime,
                                      'fromId': fromId,
                                      'recvWindow': recvWindow},
                              weight=10, decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)


def get_margin_trades_list(symbol: str,
                           decimal_mode: bool,
                           api_key: str,
                           api_secret: str,
                           isIsolated: bool = False,
                           limit: int = 1000,
                           orderId: int = None,
                           startTime: int = None,
                           endTime: int = None,
                           fromId: int = None,
                           recvWindow: int = 10000) -> list:
    """
    Get margin trades for a specific account and symbol.

    If fromId is set, it will get trades >= that fromId. Otherwise most recent trades are returned.

    Weight(IP): 10

    :param str symbol: Symbol's trades.
    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param bool isIsolated: Sets for getting isolated or not isolated trades. Default is false.
    :param int limit: Default 500; max 1000.
    :param fromId: TradeId to fetch from. Default gets most recent trades. If fromId is set, it will get id >= that fromId. Otherwise,
       most recent trades are returned.
    :param int endTime: Optional.
    :param int startTime: Optional.
    :param int orderId: This can only be used in combination with symbol.
    :param int recvWindow: The value cannot be greater than 60000
    :return list: A list.

    Example:

           .. code-block::

                     [
                        {
                            "commission": "0.00006000",
                            "commissionAsset": "BTC",
                            "id": 34,
                            "isBestMatch": true,
                            "isBuyer": false,
                            "isMaker": false,
                            "orderId": 39324,
                            "price": "0.02000000",
                            "qty": "3.00000000",
                            "symbol": "BNBBTC",
                            "isIsolated": false,
                            "time": 1561973357171
                        }
                    ]
    """

    endpoint = '/sapi/v1/margin/myTrades'

    return api_raw_signed_get(endpoint=endpoint,
                              params={'limit': limit,
                                      'isIsolated': isIsolated,
                                      'symbol': symbol,
                                      'orderId': orderId,
                                      'startTime': startTime,
                                      'endTime': endTime,
                                      'fromId': fromId,
                                      'recvWindow': recvWindow},
                              weight=10, decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)


############
# balances #
############


def get_spot_account_info(decimal_mode: bool,
                          api_key: str,
                          api_secret: str,
                          recvWindow: int = 10000) -> dict:
    """
    Get current account information.

    Weight(IP): 10

    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param int recvWindow: The value cannot be greater than 60000
    :return dict: A dictionary with data.

    Example:

        .. code-block::

           {
              "makerCommission": 15,
              "takerCommission": 15,
              "buyerCommission": 0,
              "sellerCommission": 0,
              "canTrade": true,
              "canWithdraw": true,
              "canDeposit": true,
              "updateTime": 123456789,
              "accountType": "SPOT",
              "balances": [
                {"asset": "BTC", "free": "4723846.89208129","locked": "0.00000000"},
                {"asset": "LTC","free": "4763368.68006011","locked": "0.00000000"}
              ],
              "permissions": ["SPOT"]
           }

    """
    endpoint = '/api/v3/account'
    return api_raw_signed_get(endpoint=endpoint,
                              params={'recvWindow': recvWindow},
                              weight=10,
                              decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)


def get_spot_free_balances(decimal_mode: bool,
                           api_key: str,
                           api_secret: str,
                           data_dic: dict = None
                           ) -> dict:
    """
    Parses available balances from account info.

    :param dict data_dic: If available, account info can be passed as data_dic parameter to avoid API calling.
    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :return dict: Free balances.
    """
    ret = {}
    if not data_dic:
        data_dic = get_spot_account_info(decimal_mode=decimal_mode, api_secret=api_secret, api_key=api_key)
    wallet_logger.debug(f"spot_free_balances_parsed len(get_spot_account_info()):{len(data_dic)}")
    if decimal_mode:
        for asset in data_dic['balances']:
            qty = dd(asset['free'])
            if qty:
                ret.update({asset['asset']: qty})
    else:
        for asset in data_dic['balances']:
            qty = float(asset['free'])
            if qty:
                ret.update({asset['asset']: qty})
    return ret


def get_spot_locked_balances(decimal_mode: bool,
                             api_key: str,
                             api_secret: str,
                             data_dic: dict = None) -> dict:
    """
    Parses locked in order balances from account info.

    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param dict data_dic: If available, account info can be passed as data_dic parameter to avoid API calling.
    :return dict: Locked balances.
    """
    ret = {}
    if not data_dic:
        data_dic = get_spot_account_info(decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
    wallet_logger.debug(f"spot_locked_balances_parsed get_spot_account_info: {len(data_dic)}")

    for asset in data_dic['balances']:
        qty = float(asset['locked'])
        if qty:
            ret.update({asset['asset']: qty})
    return ret


def get_coins_with_balance(decimal_mode: bool, api_key: str, api_secret: str) -> list:
    """
    Get the non-zero balances of an account, free or locked ones. Useful getting wallet value.

    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :return list:
    """
    data_dic = get_spot_account_info(decimal_mode=decimal_mode, api_secret=api_secret, api_key=api_key)
    free = get_spot_free_balances(data_dic=data_dic, decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
    locked = get_spot_locked_balances(data_dic=data_dic, decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
    free = [symbol for symbol, balance in free.items() if float(balance) > 0]
    locked = [symbol for symbol, balance in locked.items() if float(balance) > 0]
    symbols = free + locked
    return list(set(symbols))


def get_spot_balances_df(decimal_mode: bool,
                         api_key: str,
                         api_secret: str,
                         filter_empty: bool = True) -> pd.DataFrame:
    """
    Create a dataframe with the free or blocked amounts of the spot wallet. The index is the assets list.

    Example:

    .. code-block:: python

       from handlers.messages import get_spot_balances_df

       df = get_spot_balances_df()

       print(df)

                   free      locked
            asset
            BNB    0.128359     0.0
            BUSD   0.000001     0.0

    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param bool filter_empty: Discards empty quantities.
    :return pd.DataFrame: A dataframe with assets locked or free.
    """
    balances = get_spot_account_info(decimal_mode=decimal_mode,
                                     api_secret=api_secret,
                                     api_key=api_key)['balances']
    df_ = pd.DataFrame(balances)
    df_['free'] = df_['free'].apply(pd.to_numeric)
    df_['locked'] = df_['locked'].apply(pd.to_numeric)
    df_.set_index('asset', drop=True, inplace=True)
    if filter_empty:
        return df_[(df_['free'] != 0) | (df_['locked'] != 0)]
    else:
        return df_


def get_spot_balances_total_value(decimal_mode: bool,
                                  api_key: str,
                                  api_secret: str,
                                  balances_df: pd.DataFrame = None,
                                  convert_to: str = 'BUSD') -> float:
    """
    Returns total value expressed in a quote coin. Counts free and locked assets.

    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param pd.DataFrame balances_df: A BinPan balances dataframe.
    :param str convert_to: A Binance coin.
    :return float: Total quantity expressed in quote.
    """
    if decimal_mode:
        my_type = dd
    else:
        my_type = float

    if type(balances_df) != pd.DataFrame:
        balances_df = get_spot_balances_df(decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)

    prices = get_prices_dic(decimal_mode=decimal_mode)

    symbols = list(balances_df.index)
    free_values = balances_df['free'].tolist()
    locked_values = balances_df['locked'].tolist()
    total = 0

    for i in range(len(symbols)):
        coin = symbols[i]
        free = my_type(free_values[i])
        locked = my_type(locked_values[i])

        if free:
            free = convert_coin(coin=coin,
                                prices=prices,
                                convert_to=convert_to,
                                coin_qty=free,
                                decimal_mode=decimal_mode)

        if locked:
            locked = convert_coin(coin=coin,
                                  prices=prices,
                                  convert_to=convert_to,
                                  coin_qty=locked,
                                  decimal_mode=decimal_mode)
        total += my_type(free)
        total += my_type(locked)

    return total


#################
# MARGIN WALLET #
#################


def get_margin_account_details(decimal_mode: bool, api_key: str, api_secret: str) -> dict:
    """
    Query Cross Margin Account Details (USER_DATA)

    GET /sapi/v1/margin/account (HMAC SHA256)

    Weight(IP): 10

    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.

    Response:

    .. code-block::

       {
              "borrowEnabled": true,
              "marginLevel": "11.64405625",
              "totalAssetOfBtc": "6.82728457",
              "totalLiabilityOfBtc": "0.58633215",
              "totalNetAssetOfBtc": "6.24095242",
              "tradeEnabled": true,
              "transferEnabled": true,
              "userAssets": [
                  {
                      "asset": "BTC",
                      "borrowed": "0.00000000",
                      "free": "0.00499500",
                      "interest": "0.00000000",
                      "locked": "0.00000000",
                      "netAsset": "0.00499500"
                  },
                  {
                      "asset": "BNB",
                      "borrowed": "201.66666672",
                      "free": "2346.50000000",
                      "interest": "0.00000000",
                      "locked": "0.00000000",
                      "netAsset": "2144.83333328"
                  },
                  {
                      "asset": "ETH",
                      "borrowed": "0.00000000",
                      "free": "0.00000000",
                      "interest": "0.00000000",
                      "locked": "0.00000000",
                      "netAsset": "0.00000000"
                  },
                  {
                      "asset": "USDT",
                      "borrowed": "0.00000000",
                      "free": "0.00000000",
                      "interest": "0.00000000",
                      "locked": "0.00000000",
                      "netAsset": "0.00000000"
                  }
              ]
        }

    """
    margin_endpoint = '/sapi/v1/margin/account'
    ret = api_raw_signed_get(endpoint=margin_endpoint,
                             params={},
                             weight=10, decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
    return ret


###################
# margin balances #
###################

def get_margin_balances(decimal_mode: bool, api_key: str, api_secret: str) -> dict:
    """
    Collects balances in the margin account that are not null.

    :param bool decimal_mode: Fixes Decimal return type and operative.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :return dict: A dictionary with coins with balances.

    Example:

    .. code-block::

       {'BNB': {'free': 0.06,
          'locked': 0.0,
          'borrowed': 0.0,
          'interest': 0.0,
          'netAsset': 0.06},
         'BUSD': {'free': 50.0,
          'locked': 0.0,
          'borrowed': 0.0,
          'interest': 0.0,
          'netAsset': 50.0}}

    """
    margin_status = get_margin_account_details(decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
    ret = {}
    for asset in margin_status['userAssets']:
        entry = {}
        for k, v in asset.items():
            if k != 'asset':
                if decimal_mode:
                    entry[k] = dd(v)
                else:
                    entry[k] = float(v)
        if sum(entry.values()) != 0:
            ret[asset['asset']] = entry
    return ret


def get_margin_free_balances(decimal_mode: bool,
                             api_key: str,
                             api_secret: str,
                             balances: dict = None
                             ) -> dict:
    """
    Just returns free existing balances. It is optional to avoid an API call.

    :param dict balances: Returns dict with assets as keys and a float value for not null quantities.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param bool decimal_mode: Fixes Decimal return type.
    :return dict: A dict with float values.

    """
    if not balances:
        balances = get_margin_balances(decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
    ret = {}
    for k, v in balances.items():
        if 'free' in v.keys():
            if v['free']:
                ret[k] = v['free']
    return ret


def get_margin_locked_balances(decimal_mode: bool,
                               api_key: str,
                               api_secret: str,
                               balances: dict = None) -> dict:
    """
    Just returns locked existing balances. It is optional to avoid an API call.

    :param bool decimal_mode: Fixes Decimal return type.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param dict balances: Returns dict with assets as keys and a float value for not null quantities.
    :return dict: A dict with float values.

    """
    if not balances:
        balances = get_margin_balances(decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
    ret = {}
    for k, v in balances.items():
        if 'locked' in v.keys():
            if v['locked']:
                ret[k] = v['locked']
    return ret


def get_margin_borrowed_balances(decimal_mode: bool,
                                 api_key: str,
                                 api_secret: str,
                                 balances: dict = None) -> dict:
    """
    Just returns borrowed existing balances. It is optional to avoid an API call.

    :param bool decimal_mode: Fixes Decimal return type.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param dict balances: Returns dict with assets as keys and a float value for not null quantities.
    :return dict: A dict with float values.

    """
    if not balances:
        balances = get_margin_balances(decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
    ret = {}
    for k, v in balances.items():
        if 'borrowed' in v.keys():
            if v['borrowed']:
                ret[k] = v['borrowed']
    return ret


def get_margin_interest_balances(decimal_mode: bool,
                                 api_key: str,
                                 api_secret: str,
                                 balances: dict = None) -> dict:
    """
    Just returns interest existing balances. It is optional to avoid an API call.

    :param bool decimal_mode: Fixes Decimal return type.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param dict balances: Returns dict with assets as keys and a float value for not null quantities.
    :return dict: A dict with float values.

    """
    if not balances:
        balances = get_margin_balances(decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
    if not balances:
        balances = get_margin_balances(decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
    ret = {}
    for k, v in balances.items():
        if 'interest' in v.keys():
            if v['interest']:
                ret[k] = v['interest']
    return ret


def get_margin_netAsset_balances(decimal_mode: bool,
                                 api_key: str,
                                 api_secret: str,
                                 balances: dict = None):
    """
    Just returns netAsset existing balances. It is optional to avoid an API call.

    :param bool decimal_mode: Fixes Decimal return type.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param dict balances: Returns dict with assets as keys and a float value for not null quantities.
    :return dict: A dict with float values.

    """
    if not balances:
        balances = get_margin_balances(decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
    return {k: v['netAsset'] for k, v in balances.items() if v['netAsset']}


def get_margin_balances_total_value(decimal_mode: bool,
                                    api_key: str,
                                    api_secret: str,
                                    balances: dict = None,
                                    convert_to: str = 'BUSD'
                                    ) -> float or dd:
    """
    Returns total value expressed in a quote coin. Counts free, locked, borrowed and interest assets.

    :param dict balances: A BinPan balances dict. It is optional to avoid an API call.
    :param str api_key: A BinPan encrypted in BinPan's way api key. Do not use unencrypted api values, import it from  secret.py file.
    :param str api_secret: A BinPan encrypted in BinPan's way api secret. Do not use unencrypted api values, import it from  secret.py file.
    :param str convert_to: A Binance coin.
    :param bool decimal_mode: Fixes Decimal return type.
    :return float: Total quantity expressed in quote.
    """
    if decimal_mode:
        my_type = dd
    else:
        my_type = float
    prices = get_prices_dic(decimal_mode=decimal_mode)

    if not balances:
        balances = get_margin_balances(decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)

    assets = list(balances.keys())

    free_values = get_margin_free_balances(balances=balances, decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
    locked_values = get_margin_locked_balances(balances=balances, decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)
    borrowed_values = get_margin_borrowed_balances(balances=balances, decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)

    # TODO: saber si el interest tiene signo negativo
    interest_values = get_margin_interest_balances(balances=balances, decimal_mode=decimal_mode, api_key=api_key, api_secret=api_secret)

    total = 0

    for i in range(len(assets)):
        coin = assets[i]

        if coin in free_values.keys():
            free = free_values[coin]
        else:
            free = 0
        if coin in locked_values.keys():
            locked = locked_values[coin]
        else:
            locked = 0
        if coin in borrowed_values.keys():
            borrowed = borrowed_values[coin]
        else:
            borrowed = 0
        if coin in interest_values.keys():
            interest = interest_values[coin]
        else:
            interest = 0

        # TODO: saber si el interest tiene signo negativo
        total_coins = free + locked + borrowed - interest

        converted_value = convert_coin(coin=coin,
                                       prices=prices,
                                       convert_to=convert_to,
                                       coin_qty=total_coins,
                                       decimal_mode=decimal_mode)

        total += my_type(converted_value)
    return my_type(total)
