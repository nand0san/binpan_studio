"""
Statistical tests and tools.
"""

from scipy.stats import jarque_bera
import pandas as pd
import numpy as np

from .starters import is_python_version_numba_supported

try:
    if is_python_version_numba_supported():
        from numba import jit

    else:
        msg = "Cannot import numba; only Python versions >=3.7,<3.11 are supported."
        raise ImportError(msg)
except ImportError as e:
    print(e)

    # Define a no-op decorator to replace @nb.jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


def autocorrelation_coefficient(data: pd.DataFrame):
    """
    Calculate the autocorrelation coefficient of the returns in a pandas DataFrame.

    Parameters:

    data : pandas.DataFrame
        The DataFrame containing financial data in OHLC format.

    Returns:

    float
        The autocorrelation coefficient of the returns.

    Notes:

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

    Parameters:

    data : pandas.DataFrame
        The DataFrame containing financial data in OHLC format.

    Returns:

    tuple
        A tuple containing two pandas Series:
        - hourly_returns: The hourly resampled bars variance of the returns.
        - hourly_variances: The variance of the hourly variances.

    Notes:

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

    Parameters:

    data : pandas.DataFrame
        The DataFrame containing financial data in OHLC format.

    Returns:

    tuple
        A tuple containing the test statistic and p-value of the Jarque-Bera test.

    Notes:

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


############
# CALCULOS #
############


def ema_numpy(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate the Exponential Moving Average (EMA) of the given array using NumPy.

    :param arr: A NumPy array containing the input data.
    :type arr: np.ndarray
    :param window: The window size for the EMA calculation.
    :type window: int
    :return: A NumPy array containing the EMA values.
    :rtype: np.ndarray

    Example:

    .. code-block::

        import numpy as np
        arr = np.array([1.0, 2.5, 3.7, 4.2, 5.0, 6.3])
        window = 3
        ema_result = ema_numpy(arr, window)
        print(ema_result)

        [1.         1.66666667 2.61111111 3.40740741 4.27160494 5.18106996]

    """

    alpha = 2 / (window + 1)
    ema = np.zeros_like(arr)
    ema[0] = arr[0]

    for i in range(1, arr.shape[0]):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]

    return ema


def sma_numpy(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate the Simple Moving Average (SMA) of the given array using NumPy.

    :param arr: A NumPy array containing the input data.
    :type arr: np.ndarray
    :param window: The window size for the SMA calculation.
    :type window: int
    :return: A NumPy array containing the SMA values.
    :rtype: np.ndarray

    Example:

    .. code-block::

        import numpy as np
        arr = np.array([1.0, 2.5, 3.7, 4.2, 5.0, 6.3])
        window = 3
        sma_result = sma_numpy(arr, window)
        print(sma_result)

        [       nan        nan 2.4        3.46666667 4.3        5.16666667]
    """

    sma = np.convolve(arr, np.ones(window), 'valid') / window
    padding = np.full(window - 1, np.nan)
    sma_padded = np.concatenate((padding, sma))

    return sma_padded


@jit(nopython=True)
def ema_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate the Exponential Moving Average (EMA) of the given array using NumPy.

    Args:
        arr: A NumPy array containing the input data.
        window: The window size for the EMA calculation.

    Returns:
        A NumPy array containing the EMA values.

    Example:

    .. code-block::

        import numpy as np
        arr = np.array([1.0, 2.5, 3.7, 4.2, 5.0, 6.3])
        window = 3
        ema_result = ema_numpy(arr, window)
        print(ema_result)

        [1.         1.66666667 2.61111111 3.40740741 4.27160494 5.18106996]
    """
    alpha = 2 / (window + 1)
    ema = np.zeros_like(arr)
    ema[0] = arr[0]
    for i in range(1, arr.shape[0]):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
    return ema


@jit(nopython=True)
def sma_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate the Simple Moving Average (SMA) of the given array using NumPy.

    Args:
        arr: A NumPy array containing the input data.
        window: The window size for the SMA calculation.

    Returns:
        A NumPy array containing the SMA values.

    Example:

    .. code-block::

        import numpy as np
        arr = np.array([1.0, 2.5, 3.7, 4.2, 5.0, 6.3])
        window = 3
        sma_result = sma_numpy(arr, window)
        print(sma_result)

        [       nan        nan 2.4        3.46666667 4.3        5.16666667]
    """
    sma = np.empty_like(arr, dtype=np.float64)
    for i in range(arr.shape[0]):
        start = max(0, i - window + 1)
        sma[i] = np.sum(arr[start:i + 1]) / (i - start + 1)
    return sma
