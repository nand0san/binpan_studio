import numpy as np
from typing import Tuple

try:
    from numba import njit
    print("Numba imported")
except Exception as e:
    msg = "Cannot import numba. Optionally install numba for better performance: 'pip install numba'"
    print(msg)

    # noinspection PyUnusedLocal
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@njit(cache=True)
def rolling_max_with_steps_back_numba(values: np.ndarray, window: int, pct_diff: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the rolling maximum of the given array using NumPy.

    :param values: A NumPy array containing the input data.
    :param window: An integer for the window size.
    :param pct_diff: A boolean indicating whether to calculate the percentage difference.
    :return: A tuple containing the rolling maximum value and the steps back.
    """

    n = len(values)
    rolling_max = np.empty(n, dtype=np.float64)
    steps_back = np.empty(n, dtype=np.int64)

    for i in range(n):
        window_start = max(0, i - window + 1)
        window_values = values[window_start:i + 1]
        if pct_diff:
            current_max = np.max(window_values)
            rolling_max[i] = values[i] / current_max - 1 if current_max != 0 else 0
        else:
            rolling_max[i] = np.max(window_values)

        max_idx = np.where(window_values == rolling_max[i])[0][-1]
        steps_back[i] = window - 1 - (len(window_values) - max_idx - 1)

    return rolling_max, steps_back


@njit(cache=True)
def rolling_min_with_steps_back_numba(values: np.ndarray, window: int, pct_diff: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the rolling minimum of the given array using NumPy.

    :param values: A NumPy array containing the input data.
    :param window: An integer for the window size.
    :param pct_diff: A boolean indicating whether to calculate the percentage difference.
    :return: A tuple containing the rolling minimum value and the steps back.
    """
    n = len(values)
    rolling_min = np.empty(n, dtype=np.float64)
    steps_back = np.empty(n, dtype=np.int64)

    for i in range(n):
        window_start = max(0, i - window + 1)
        window_values = values[window_start:i + 1]
        if pct_diff:
            current_min = np.min(window_values)
            rolling_min[i] = values[i] / current_min - 1 if current_min != 0 else 0
        else:
            rolling_min[i] = np.min(window_values)

        min_idx = np.where(window_values == rolling_min[i])[0][-1]
        steps_back[i] = window - 1 - (len(window_values) - min_idx - 1)

    return rolling_min, steps_back


@njit(cache=True)
def ema_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate the Exponential Moving Average (EMA) of the given array using NumPy.

    Note: Results are identical to the pandas-ta implementation.

    :param arr: A NumPy array containing the input data.
    :param window: The window size for the EMA calculation.
    :return: A NumPy array containing the EMA values.

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


@njit(cache=True)
def rma_numba(values: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate the Rolling Moving Average (RMA) of the given array using NumPy.

    :param values: A NumPy array containing the input data.
    :param window: An integer for the window size.
    :return: A value containing the RMA.
    """
    alpha = 1.0 / window
    scale = 1.0 - alpha
    n = len(values)
    avg = np.empty(n)
    avg[0] = values[0]
    for i in range(1, n):
        avg[i] = alpha * values[i] + scale * avg[i - 1]
    return avg


@njit(cache=True)
def rsi_numba(close: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate the Relative Strength Index (RSI) of the given array using NumPy.

    Note: Results are identical to the pandas-ta implementation.

    :param close: A NumPy array containing the closing prices.
    :param window: An integer for the window size.
    :return: A NumPy array containing the RSI values.
    """
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = rma_numba(values=gain, window=window)
    avg_loss = rma_numba(values=loss, window=window)

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Convertir NaN a un array de NumPy y luego concatenar
    nan_array = np.array([np.nan])
    return np.concatenate((nan_array, rsi))


@njit(cache=True)
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


@njit(cache=True)
def close_support_log_numba(close: np.ndarray, support: np.ndarray) -> np.ndarray:
    """
    Calculate the logarithmic ratio between the closing price and the closest support level.

    :param close: A NumPy array containing the closing prices.
    :param support: A NumPy array containing the support levels.
    :return: A NumPy array containing the logarithmic ratio.
    """
    support = np.sort(support)
    indices = np.searchsorted(support, close, side='right') - 1
    # Ajustamos los índices para asegurarnos de que no sean negativos
    indices = np.clip(indices, 0, len(support) - 1)
    closest_support = support[indices]
    # Usamos el valor de cierre cuando no hay soporte más cercano inferior
    closest_support = np.where(close == support[indices], close, closest_support)
    # Evitar división por cero o logaritmo de un número negativo
    closest_support = np.maximum(closest_support, 1e-9)
    return np.log(close / closest_support)


@njit(cache=True)
def close_resistance_log_numba(close: np.ndarray, resistance: np.ndarray) -> np.ndarray:
    """
    Calculate the logarithmic ratio between the closest resistance level and the closing price.

    :param close: A NumPy array containing the closing prices.
    :param resistance: A NumPy array containing the resistance levels.
    :return: A NumPy array containing the logarithmic ratio.
    """
    resistance = np.sort(resistance)
    indices = np.searchsorted(resistance, close, side='left')
    # Asegurarnos de que los índices no superen el máximo índice válido
    indices = np.minimum(indices, len(resistance) - 1)
    closest_resistance = resistance[indices]
    # Reemplazar los valores donde no hay resistencia superior con el valor de cierre
    closest_resistance = np.where(indices == len(resistance), close, closest_resistance)
    # Evitar logaritmo de un número negativo o cero
    closest_resistance = np.maximum(closest_resistance, 1e-9)
    # Calcular la diferencia logarítmica
    return np.log(np.where(closest_resistance == close, 1, closest_resistance / close))
