import numpy as np
from typing import Tuple, List
from numba import njit, prange


@njit(cache=True)
def rolling_max_with_steps_back_numba(values, window, pct_diff) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the rolling maximum and the number of steps back to the rolling maximum within a moving window.

    :param values: A 1D NumPy array containing the input data.
    :param window: An integer specifying the size of the moving window.
    :param pct_diff: A boolean to determine if the output should be the percentage difference
                     between the current value and the rolling maximum. If False, the rolling maximum is returned.
    :return: A tuple of two 1D NumPy arrays. The first array is the rolling maximum or the percentage difference
             (based on `pct_diff`). The second array contains the number of steps back to the rolling maximum.
    """
    n = len(values)
    rolling_max = np.full(n, np.nan)  # Inicializar con NaN
    steps_back = np.full(n, -1)  # Inicializar con -1

    for i in range(window - 1, n):
        window_values = values[i - window + 1:i + 1]

        # Ignorar los NaNs en la ventana actual
        valid_values = window_values[~np.isnan(window_values)]

        if len(valid_values) > 0:
            current_max = np.max(valid_values)
            rolling_max[i] = values[i] / current_max - 1 if pct_diff and current_max != 0 else current_max
            max_idx = np.where(window_values == current_max)[0][-1]
            steps_back[i] = window - 1 - (len(window_values) - max_idx - 1)

    return rolling_max, steps_back


@njit(cache=True)
def rolling_max_with_steps_back_last_value_numba(values: np.ndarray, window: int, pct_diff: bool) -> Tuple[float, int]:
    """
    Calculate the rolling maximum and the steps back to the maximum for the last value in the input array.

    :param values: A 1D NumPy array containing the input data.
    :param window: An integer specifying the size of the moving window.
    :param pct_diff: A boolean to determine if the output should be the percentage difference
                     between the last value and its rolling maximum. If False, the rolling maximum is returned.
    :return: A tuple containing the rolling maximum or the percentage difference (based on `pct_diff`) for the last value,
             and the steps back to the rolling maximum for the last value.
    """

    n = len(values)

    # Asegurarse de que la ventana no sea mayor que el tamaño de 'values'
    effective_window = min(window, n)

    # Obtener los valores de la última ventana
    window_values = values[-effective_window:]

    # Ignorar los NaNs en la ventana actual
    valid_values = window_values[~np.isnan(window_values)]

    if len(valid_values) > 0:
        current_max = np.max(valid_values)
        last_value_max = values[-1] / current_max - 1 if pct_diff and current_max != 0 else current_max
        max_idx = np.where(window_values == current_max)[0][-1]
        steps_back = effective_window - 1 - (len(window_values) - max_idx - 1)
    else:
        last_value_max = np.nan
        steps_back = -1

    return last_value_max, steps_back


@njit(cache=True)
def rolling_min_with_steps_back_numba(values, window, pct_diff) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the rolling minimum and the number of steps back to the rolling minimum within a moving window.

    :param values: A 1D NumPy array containing the input data.
    :param window: An integer specifying the size of the moving window.
    :param pct_diff: A boolean to determine if the output should be the percentage difference
                     between the current value and the rolling minimum. If False, the rolling minimum is returned.
    :return: A tuple of two 1D NumPy arrays. The first array is the rolling minimum or the percentage difference
             (based on `pct_diff`). The second array contains the number of steps back to the rolling minimum.
    """
    n = len(values)
    rolling_min = np.full(n, np.nan)  # Inicializar con NaN
    steps_back = np.full(n, -1)  # Inicializar con -1

    for i in range(window - 1, n):
        window_values = values[i - window + 1:i + 1]

        # Ignorar los NaNs en la ventana actual
        valid_values = window_values[~np.isnan(window_values)]

        if len(valid_values) > 0:
            current_min = np.min(valid_values)
            rolling_min[i] = values[i] / current_min - 1 if pct_diff and current_min != 0 else current_min
            min_idx = np.where(window_values == current_min)[0][-1]
            steps_back[i] = window - 1 - (len(window_values) - min_idx - 1)

    return rolling_min, steps_back


@njit(cache=True)
def rolling_min_with_steps_back_last_value_numba(values: np.ndarray, window: int, pct_diff: bool) -> Tuple[float, int]:
    """
    Calculate the rolling minimum for the last value in the given array using NumPy.

    :param values: A NumPy array containing the input data.
    :param window: An integer for the window size.
    :param pct_diff: A boolean indicating whether to calculate the percentage difference.
    :return: A tuple containing the rolling minimum value and the steps back for the last value.
    """
    n = len(values)

    # Asegurarse de que la ventana no sea mayor que el tamaño de 'values'
    effective_window = min(window, n)

    # Obtener los valores de la última ventana
    window_values = values[-effective_window:]

    # Ignorar los NaNs en la ventana actual
    valid_values = window_values[~np.isnan(window_values)]

    if len(valid_values) > 0:
        current_min = np.min(valid_values)
        last_value_min = values[-1] / current_min - 1 if pct_diff and current_min != 0 else current_min
        min_idx = np.where(window_values == current_min)[0][-1]
        steps_back = effective_window - 1 - (len(window_values) - min_idx - 1)
    else:
        last_value_min = np.nan
        steps_back = -1

    return last_value_min, steps_back


@njit(cache=True)
def ema_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate the Exponential Moving Average (EMA) of the given array.

    :param arr: A 1D NumPy array containing the input data.
    :param window: An integer specifying the size of the moving window for EMA calculation.
    :return: A 1D NumPy array containing the EMA values.

    Note: The EMA is calculated using the formula `EMA[today] = (Value[today] * (2 / (1 + window))) + (EMA[yesterday] * (1 - (2 / (1 + window))))`.
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
    Calculate the Rolling Moving Average (RMA) of the given array.

    :param values: A 1D NumPy array containing the input data.
    :param window: An integer specifying the size of the moving window for RMA calculation.
    :return: A 1D NumPy array containing the RMA values.

    Note: The RMA is a type of moving average that gives more weight to recent data points, similar to the EMA.
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
    Calculate the Relative Strength Index (RSI) for an array of closing prices.

    :param close: A 1D NumPy array containing the closing prices.
    :param window: An integer specifying the size of the moving window for RSI calculation.
    :return: A 1D NumPy array containing the RSI values.

    Note: The RSI is calculated using the formula `RSI = 100 - (100 / (1 + (Average Gain / Average Loss)))`.
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
def close_support_dynamic_numba(close: np.ndarray, supports: np.ndarray) -> np.ndarray:
    """
    Calculate the logarithmic ratio between the closing price and the closest support level
    for each point in time, where the support levels can change over time.

    :param close: A NumPy array containing the closing prices.
    :param supports: A 2D NumPy array where each row contains the support levels at a given time.
    :return: A NumPy array containing the logarithmic ratio for each point in time.
    """
    log_ratios = np.zeros(close.shape[0])

    for i in range(close.shape[0]):
        # Obtener los soportes para el tiempo actual y ordenarlos
        current_supports = supports[i, :]
        current_supports = current_supports[~np.isnan(current_supports)]  # Eliminar NaNs
        current_supports = np.sort(current_supports)

        # Encontrar el soporte más cercano por debajo del precio de cierre actual
        if current_supports.size > 0:
            index = np.searchsorted(current_supports, close[i], side='right') - 1
            index = 0 if index < 0 else min(index, len(current_supports) - 1)
            closest_support = current_supports[index]
            # Usamos el valor de cierre cuando no hay soporte más cercano inferior
            closest_support = close[i] if close[i] < current_supports[0] else closest_support
        else:
            # Si no hay soportes, usar el valor de cierre para evitar división por cero
            closest_support = close[i]

        # Evitar división por cero o logaritmo de un número negativo
        # noinspection PyTypeChecker
        closest_support = max(closest_support, 1e-9)
        log_ratios[i] = np.log(close[i] / closest_support)

    return log_ratios


@njit(cache=True)
def close_support_log_single_numba(close: np.ndarray, support: np.ndarray) -> np.float64:
    """
    Calculate the logarithmic ratio between a single closing price and the closest support level to it.

    :param close: A single closing price (float).
    :param support: A NumPy array containing the support levels.
    :return: The logarithmic ratio for the given closing price.
    """
    # support = np.sort(support)
    index = np.searchsorted(support, close, side='right') - 1
    index = max(index, 0)  # Asegurar que el índice no sea negativo
    closest_support = support[index]
    closest_support = close if close == support[index] else closest_support
    # noinspection PyTypeChecker
    closest_support = max(closest_support, 1e-9)  # Evitar división por cero o logaritmo de un número negativo
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


@njit(cache=True)
def close_resistance_dynamic_numba(close: np.ndarray, resistances: np.ndarray) -> np.ndarray:
    """
    Calculate the logarithmic ratio between the closest resistance level and the closing price
    for each point in time, where the resistance levels can change over time.

    :param close: A NumPy array containing the closing prices.
    :param resistances: A 2D NumPy array where each row contains the resistance levels at a given time.
    :return: A NumPy array containing the logarithmic ratio for each point in time.
    """
    log_ratios = np.zeros(close.shape[0])

    for i in range(close.shape[0]):
        # Obtener las resistencias para el tiempo actual y ordenarlas
        current_resistances = resistances[i, :]
        current_resistances = current_resistances[~np.isnan(current_resistances)]  # Eliminar NaNs
        current_resistances = np.sort(current_resistances)

        # Encontrar la resistencia más cercana por encima del precio de cierre actual
        if current_resistances.size > 0:
            indices = np.searchsorted(current_resistances, close[i], side='left')
            indices = np.minimum(indices, len(current_resistances) - 1)
            closest_resistance = current_resistances[indices]
            # Reemplazar los valores donde no hay resistencia superior con el valor de cierre
            closest_resistance = close[i] if close[i] > current_resistances[-1] else closest_resistance
        else:
            # Si no hay resistencias, usar el valor de cierre para evitar división por cero
            closest_resistance = close[i]

        # Evitar logaritmo de un número negativo o cero
        closest_resistance = np.maximum(closest_resistance, 1e-9)
        # Calcular la diferencia logarítmica
        log_ratios[i] = np.log(np.where(closest_resistance == close[i], 1, closest_resistance / close[i]))

    return log_ratios


@njit(cache=True)
def close_resistance_log_single_numba(close: np.ndarray, resistance: np.ndarray) -> np.float64:
    """
    Calculate the logarithmic ratio between the closest resistance level and a single closing price.

    :param close: A single closing price (float).
    :param resistance: A NumPy array containing the resistance levels.
    :return: The logarithmic ratio for the given closing price.
    """
    # resistance = np.sort(resistance)
    index_ = np.searchsorted(resistance, close, side='left')
    # noinspection PyTypeChecker
    index = min(index_, len(resistance) - 1)  # Asegurar que el índice no sea mayor que el máximo índice válido
    closest_resistance = resistance[index]
    # noinspection PyTypeChecker
    closest_resistance = max(closest_resistance, 1e-9)  # Evitar logaritmo de un número negativo o cero

    # Calcular la diferencia logarítmica
    return np.log(max(closest_resistance / close, 1))


@njit(parallel=True)
def calculate_ema_relations(arr: np.ndarray, windows: np.ndarray, epsilon: float = 1e-8) -> List[np.ndarray]:
    """
    Calcula las relaciones de media móvil exponencial (EMA) para una serie de valores, utilizando un conjunto de ventanas especificadas.

    La función utiliza Numba para la optimización del cálculo, especialmente efectiva en grandes conjuntos de datos. Para cada ventana en
    'windows', calcula la EMA de la serie 'arr'. Luego, ajusta los valores de EMA para evitar divisiones por cero, usando 'epsilon'. Finalmente, devuelve una lista de arrays, cada uno representando la serie 'arr' dividida por su respectiva EMA calculada.

    :param arr : Array de valores para el cálculo de EMA.
    :param windows : Array de ventanas para el cálculo de EMA.
    :param epsilon : Valor de ajuste para evitar divisiones por cero.
    :return : Lista de arrays, cada uno representando la serie 'arr' dividida por su respectiva EMA calculada.

    """
    results = []
    for i in prange(len(windows)):
        window = windows[i]
        current_ema = ema_numba(arr, window)
        current_ema = np.where(current_ema == 0, epsilon, current_ema)
        result = arr / current_ema
        results.append(result)
    return results
