import pandas as pd

from handlers.files import get_encoded_secrets
from handlers.market import convert_coin
from handlers.wallet import get_spot_balances_df, daily_account_snapshot, convert_str_date_to_ms, get_margin_balances


class Wallet(object):
    """
    Wallet is a BinPan Class that can give information about balances, in Spot or Margin trading.

    Also can show snapshots of the account status days ago, or using timestamps.

    """

    def __init__(self, time_zone='UTC', snapshot_days: int = 30):
        self.api_key, self.api_secret = get_encoded_secrets()

        self.time_zone = time_zone
        self.spot = self.update_spot()
        self.spot_snapshot = None
        self.spot_startTime = None
        self.spot_endTime = None
        self.spot_requested_days = snapshot_days

        self.margin = self.update_margin()
        self.margin_snapshot = None
        self.margin_startTime = None
        self.margin_endTime = None
        self.margin_requested_days = snapshot_days

    def update_spot(self, decimal_mode=False):
        """
        Updates balances in the class object.
        :param bool decimal_mode: Use of decimal objects instead of float.
        :return dict: Wallet dictionary
        """
        self.spot = get_spot_balances_df(decimal_mode=decimal_mode, api_key=self.api_key, api_secret=self.api_secret)
        return self.spot

    def spot_snapshot(self, startTime: int or str = None, endTime: int or str = None, snapshot_days=30, time_zone=None):
        """
        Updates spot wallet snapshot.

        :param int or str startTime: Can be integer timestamp in milliseconds or formatted string: 2022-05-11 06:45:42
        :param int or str endTime: Can be integer timestamp in milliseconds or formatted string: 2022-05-11 06:45:42
        :param int snapshot_days: Days to look if not start time or endtime passed.
        :param str time_zone: A time zone for time index conversion. Example: "Europe/Madrid"
        :return pd.DataFrame: Spot wallet snapshot for the time period requested.
        """
        if time_zone:
            self.time_zone = time_zone

        self.spot_startTime = startTime
        self.spot_snapshot = daily_account_snapshot(account_type='SPOT', startTime=convert_str_date_to_ms(date=startTime,
                                                                                                          time_zone=self.time_zone),
                                                    endTime=convert_str_date_to_ms(
                                                        date=endTime,
                                                        time_zone=self.time_zone),
                                                    limit=snapshot_days,
                                                    time_zone=self.time_zone, decimal_mode=False, api_key=self.api_key,
                                                    api_secret=self.api_secret)
        self.spot_endTime = endTime
        self.spot_requested_days = snapshot_days

        return self.spot

    def update_margin(self, decimal_mode=False):
        """
        Updates balances in the wallet class object.

        :param bool decimal_mode: Use of decimal objects instead of float.
        :return dict: Wallet dictionary
        """
        my_margin = get_margin_balances(decimal_mode=decimal_mode, api_key=self.api_key, api_secret=self.api_secret)
        self.margin = pd.DataFrame(my_margin).T
        self.margin.index.name = 'asset'
        return self.margin

    def margin_snapshot(self, startTime: int or str = None, endTime: int or str = None, snapshot_days=30, time_zone=None):
        """
        Updates margin wallet snapshot.

        :param int or str startTime: Can be integer timestamp in milliseconds or formatted string: 2022-05-11 06:45:42
        :param int or str endTime: Can be integer timestamp in milliseconds or formatted string: 2022-05-11 06:45:42
        :param int snapshot_days: Days to look if not start time or endtime passed.
        :param str time_zone: A time zone for time index conversion. Example: "Europe/Madrid"
        :return pd.DataFrame: Spot wallet snapshot for the time period requested.
        """
        if time_zone:
            self.time_zone = time_zone

        self.spot = daily_account_snapshot(account_type='MARGIN', startTime=convert_str_date_to_ms(date=startTime,
                                                                                                   time_zone=self.time_zone),
                                           endTime=convert_str_date_to_ms(
                                               date=endTime,
                                               time_zone=self.time_zone),
                                           limit=snapshot_days,
                                           time_zone=self.time_zone, decimal_mode=False, api_key=self.api_key, api_secret=self.api_secret)
        self.margin_startTime = startTime
        self.margin_endTime = endTime
        self.margin_requested_days = snapshot_days
        return self.margin

    def spot_wallet_performance(self, decimal_mode: bool, startTime=None, endTime=None, days: int = 30, convert_to: str = 'BUSD'):
        """
        Calculate difference between current wallet not locked values and days before.
        :param bool decimal_mode: Fixes Decimal return type and operative.
        :param int or str startTime: Can be integer timestamp in milliseconds or formatted string: 2022-05-11 06:45:42
        :param int or str endTime: Can be integer timestamp in milliseconds or formatted string: 2022-05-11 06:45:42
        :param int days: Days to compare balances.
        :param str convert_to: Converts balances to a coin.
        :return float: Value increase or decrease with current value of convert_to coin.
        """
        if days != self.spot_requested_days or startTime != self.spot_startTime or endTime != self.spot_endTime:
            self.spot = daily_account_snapshot(account_type='SPOT', startTime=convert_str_date_to_ms(date=startTime,
                                                                                                     time_zone=self.time_zone),
                                               endTime=convert_str_date_to_ms(
                                                   date=endTime,
                                                   time_zone=self.time_zone), limit=days,
                                               time_zone=self.time_zone, decimal_mode=False, api_key=self.api_key,
                                               api_secret=self.api_secret)
            self.spot_startTime = startTime
            self.spot_endTime = endTime
            self.spot_requested_days = days

        if not self.spot.empty:
            totalAssetOfBtc = self.spot['totalAssetOfBtc'].tolist()
            performance = totalAssetOfBtc[-1] - totalAssetOfBtc[0]
            if convert_to == 'BTC':
                return performance
            else:
                return convert_coin(coin='BTC', convert_to=convert_to, coin_qty=performance, decimal_mode=decimal_mode)
        else:
            return 0

    def margin_wallet_performance(self, decimal_mode: bool, startTime=None, endTime=None, days: int = 30, convert_to: str = 'BUSD'):
        """
        Calculate difference between current wallet not locked values and days before.
        :param bool decimal_mode: Fixes Decimal return type and operative.
        :param int or str startTime: Can be integer timestamp in milliseconds or formatted string: 2022-05-11 06:45:42
        :param int or str endTime: Can be integer timestamp in milliseconds or formatted string: 2022-05-11 06:45:42
        :param int days: Days to compare balances.
        :param str convert_to: Converts balances to a coin.
        :return float: Value increase or decrease with current value of convert_to coin.
        """
        if days != self.margin_requested_days or startTime != self.margin_startTime or endTime != self.margin_endTime:
            self.margin = daily_account_snapshot(account_type='MARGIN', startTime=convert_str_date_to_ms(date=startTime,
                                                                                                         time_zone=self.time_zone),
                                                 endTime=convert_str_date_to_ms(
                                                     date=endTime,
                                                     time_zone=self.time_zone),
                                                 limit=days,
                                                 time_zone=self.time_zone, decimal_mode=False, api_key=self.api_key,
                                                 api_secret=self.api_secret)
            self.margin_startTime = startTime
            self.margin_endTime = endTime
            self.margin_requested_days = days

        if not self.margin.empty:
            totalAssetOfBtc = self.margin['totalAssetOfBtc'].tolist()
            performance = totalAssetOfBtc[-1] - totalAssetOfBtc[0]
            if convert_to == 'BTC':
                return performance
            else:
                return convert_coin(coin='BTC', convert_to=convert_to, coin_qty=performance, decimal_mode=decimal_mode)
        else:
            return 0
