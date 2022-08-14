import pandas as pd

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


def daily_account_snapshot(account_type: str = 'SPOT',
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
                             weight=2400)
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


def assets_convertible_dust() -> dict:
    """
    Assets dust that can be converted to BNB.
    Weight(IP): 1

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
        return api_raw_signed_post(endpoint='/sapi/v1/asset/dust-btc', weight=1)
    else:
        wallet_logger.warning("Canceled!")


##########
# trades #
##########


def get_spot_trades_list(symbol: str,
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
                              weight=10)


############
# balances #
############


def get_spot_account_info(recvWindow: int = 10000) -> dict:
    """
    Get current account information.

    Weight(IP): 10

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
                              weight=10)


def spot_free_balances_parsed(data_dic: dict = None) -> dict:
    """
    Parses available balances from account info.

    :param dict data_dic: If available, account info can be passed as data_dic parameter to avoid API calling.
    :return dict: Free balances.
    """
    ret = {}
    if not data_dic:
        data_dic = get_spot_account_info()
    wallet_logger.debug(f"spot_free_balances_parsed len(get_spot_account_info()):{len(data_dic)}")
    for asset in data_dic['balances']:
        qty = float(asset['free'])
        if qty:
            ret.update({asset['asset']: qty})
    return ret


def spot_locked_balances_parsed(data_dic: dict = None) -> dict:
    """
    Parses locked in order balances from account info.

    :param dict data_dic: If available, account info can be passed as data_dic parameter to avoid API calling.
    :return dict: Locked balances.
    """
    ret = {}
    if not data_dic:
        data_dic = get_spot_account_info()
    wallet_logger.debug(f"spot_locked_balances_parsed get_spot_account_info: {len(data_dic)}")

    for asset in data_dic['balances']:
        qty = float(asset['locked'])
        if qty:
            ret.update({asset['asset']: qty})
    return ret


def get_coins_with_balance() -> list:
    """
    Get the non-zero balances of an account, free or locked ones. Useful getting wallet value.

    :return list:
    """
    data_dic = get_spot_account_info()
    free = spot_free_balances_parsed(data_dic=data_dic)
    locked = spot_locked_balances_parsed(data_dic=data_dic)
    free = [symbol for symbol, balance in free.items() if float(balance) > 0]
    locked = [symbol for symbol, balance in locked.items() if float(balance) > 0]
    symbols = free + locked
    return list(set(symbols))


def get_spot_balances_df(filter_empty: bool = True) -> pd.DataFrame:
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


    :param bool filter_empty: Discards empty quantities.
    :return pd.DataFrame: A dataframe with assets locked or free.
    """
    balances = get_spot_account_info()['balances']
    df_ = pd.DataFrame(balances)
    df_['free'] = df_['free'].apply(pd.to_numeric)
    df_['locked'] = df_['locked'].apply(pd.to_numeric)
    df_.set_index('asset', drop=True, inplace=True)
    if filter_empty:
        return df_[(df_['free'] != 0) | (df_['locked'] != 0)]
    else:
        return df_


def get_spot_balances_total_value(balances_df: pd.DataFrame = None,
                                  convert_to: str = 'BUSD') -> float:
    """
    Returns total value expressed in a quote coin. Counts free and locked assets.
    
    :param pd.DataFrame balances_df: A BinPan balances dataframe. 
    :param str convert_to: A Binance coin. 
    :return float: Total quantity expressed in quote. 
    """
    if type(balances_df) != pd.DataFrame:
        balances_df = get_spot_balances_df()

    prices = get_prices_dic()

    symbols = list(balances_df.index)
    free_values = balances_df['free'].tolist()
    locked_values = balances_df['locked'].tolist()
    total = 0

    for i in range(len(symbols)):
        coin = symbols[i]
        free = float(free_values[i])
        locked = float(locked_values[i])

        if free:
            free = convert_coin(coin=coin,
                                prices=prices,
                                convert_to=convert_to,
                                coin_qty=free)

        if locked:
            locked = convert_coin(coin=coin,
                                  prices=prices,
                                  convert_to=convert_to,
                                  coin_qty=locked)
        total += float(free)
        total += float(locked)

    return total


#################
# MARGIN WALLET #
#################


def get_margin_account_details() -> dict:
    """
    Query Cross Margin Account Details (USER_DATA)
    GET /sapi/v1/margin/account (HMAC SHA256)

    Weight(IP): 10

    Parameters:

    Name	Type	Mandatory	Description
    recvWindow	LONG	NO	The value cannot be greater than 60000
    timestamp	LONG	YES

    Response:

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
    margin_endpoint='/sapi/v1/margin/account'
    ret = api_raw_signed_get(endpoint=margin_endpoint,
                             params={},
                             weight=10)
    return ret


def get_margin_balances():
    """Balances en la cuenta margin no nulos"""
    margin_status = get_margin_account_details()
    ret = {}
    for asset in margin_status['userAssets']:
        entry = {}
        for k, v in asset.items():
            if k != 'asset':
                entry[k] = float(v)
        if sum(entry.values()) != 0:
            ret[asset['asset']] = entry
    return ret


def get_margin_free_balances(balances: dict = None):
    if not balances:
        balances = get_margin_balances()
    return {k: v['free'] for k, v in balances.items() if v['free']}


def get_margin_locked_balances(balances: dict = None):
    if not balances:
        balances = get_margin_balances()
    return {k: v['locked'] for k, v in balances.items() if v['locked']}


def get_margin_borrowed_balances(balances: dict = None):
    if not balances:
        balances = get_margin_balances()
    return {k: v['borrowed'] for k, v in balances.items() if v['borrowed']}


def get_margin_interest_balances(balances: dict = None):
    if not balances:
        balances = get_margin_balances()
    return {k: v['interest'] for k, v in balances.items() if v['interest']}


def get_margin_netAsset_balances(balances: dict = None):
    if not balances:
        balances = get_margin_balances()
    return {k: v['netAsset'] for k, v in balances.items() if v['netAsset']}
