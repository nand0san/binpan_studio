"""
Statistical tests and tools.
"""

from scipy.stats import jarque_bera
import pandas as pd
import numpy as np


def autocorrelation_coefficient(data: pd.DataFrame):
    """
    Calculate the autocorrelation coefficient of the returns in a pandas DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing financial data in OHLC format.

    Returns
    -------
    float
        The autocorrelation coefficient of the returns.

    Notes
    -----
    The autocorrelation coefficient measures the strength of the linear relationship between
    the returns and their lagged values. A value of zero indicates no correlation, while a value
    of one indicates perfect positive correlation and a value of negative one indicates perfect
    negative correlation.

    """
    df = data.copy(deep=True)

    # calculate the returns
    df['Return'] = df['Close'].pct_change()

    corr = df['Return'].autocorr()

    return corr


def compute_variances(data: pd.DataFrame):
    """
    Compute the hourly resampled bars variance and variance of variances for a given OHLC DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing financial data in OHLC format.

    Returns
    -------
    tuple
        A tuple containing two pandas Series:
        - hourly_returns: The hourly resampled bars variance of the returns.
        - hourly_variances: The variance of the hourly variances.

    Notes
    -----
    This function calculates the returns as the percentage change in the 'Close' column of the input
    DataFrame, and then groups the data by hour using the `pd.Grouper` method. The hourly resampled bars
    variance of the returns is then calculated for each hour, and the variance of these variances is
    calculated and returned as `hourly_variances`.

    """
    df = data.copy(deep=True)

    # calculate the returns
    df['Return'] = df['Close'].pct_change()

    # group the data by hour and calculate the variance of returns for each hour
    hourly_returns = df['Return'].groupby(pd.Grouper(freq='H')).var()

    # calculate the variance of the hourly variances
    hourly_variances = hourly_returns.var()

    return hourly_returns, hourly_variances


def jarque_bera_test(data):
    """
    Perform the Jarque-Bera test for normality on the returns of an OHLC DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing financial data in OHLC format.

    Returns
    -------
    tuple
        A tuple containing the test statistic and p-value of the Jarque-Bera test.

    Notes
    -----
    This function first calculates the returns as the percentage change in the 'Close' column of the input
    DataFrame, and then drops the first row, which will have a NaN return. Finally, the Jarque-Bera test for
    normality is performed on the returns using the `jarque_bera` function from the `scipy.stats` module.
    The test statistic and p-value are returned as a tuple.

    The null hypothesis of the Jarque-Bera test is that the sample of data being tested is drawn from a normal
    distribution. Therefore, a small p-value (e.g., less than 0.05) indicates that the null hypothesis is rejected
    and the data is not normally distributed.
    """
    df = data.copy(deep=True)

    # calculate the returns
    df['Return'] = df['Close'].pct_change()

    # drop the first row, which will have a NaN return
    df = df.dropna()

    # perform the Jarque-Bera test on the returns
    test = jarque_bera(df['Return'])
    return test
