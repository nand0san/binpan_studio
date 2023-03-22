Statistical Tests and Tools
===========================

This module provides statistical tests and tools for financial data analysis.

This module can be imported:

.. code-block::

   from handlers import stat_tests

.. automodule:: handlers.stat_tests


autocorrelation_coefficient
---------------------------

.. autofunction:: handlers.stat_tests.autocorrelation_coefficient

compute_variances
-----------------

.. autofunction:: handlers.stat_tests.compute_variances

jarque_bera_test
-----------------

.. autofunction:: handlers.stat_tests.jarque_bera_test


Notes
-----

The `autocorrelation_coefficient` function calculates the autocorrelation coefficient of the returns in a pandas DataFrame.

The `compute_variances` function calculates the hourly resampled bars variance and variance of variances for a given OHLC DataFrame.

The `jarque_bera_test` function performs the Jarque-Bera test for normality on the returns of an OHLC DataFrame.

The null hypothesis of the Jarque-Bera test is that the sample of data being tested is drawn from a normal distribution. Therefore, a small p-value (e.g., less than 0.05) indicates that the null hypothesis is rejected and the data is not normally distributed.


Calculations
============

ema_numpy
---------

.. autofunction:: handlers.stat_tests.ema_numpy

sma_numpy
---------

.. autofunction:: handlers.stat_tests.sma_numpy

ema_numba
---------

.. autofunction:: handlers.stat_tests.ema_numba

sma_numba
---------

.. autofunction:: handlers.stat_tests.sma_numba

Notes
-----

The ema_numpy and sma_numpy functions use numpy library for the calculations, and are suitable for small to medium-sized datasets.

The ema_numba and sma_numba functions use numba library for the calculations, and are optimized for larger datasets.

All functions require the input data to be a 1D numpy array of numerical values.
