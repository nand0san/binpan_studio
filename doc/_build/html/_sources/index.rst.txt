
Welcome to BinPan's documentation!
==================================

BinPan is a Python wrapper for Binance API.

BinPan can make plots easily and some API requests in the same object.

The objective of this module is to have a fast tool to collect and handle data from the Binance API
in an agile way.

It is intended to be useful in Jupyter Notebooks or even the python console, but it can be used in
many other ways.

BinPan manages symbol objects that can:

- get candles with time zone and indexing options.
- get trades.
- calculate technical indicators.
- plot candles, histograms, indicators, etc in a very simple and beautiful way.

Hope you find it useful breaking the market!!!


Installation
------------

.. code-block:: bash

   pip install binpan

Usage
-----

Importing just like this:

.. code-block::

    import binpan

    btcusdt = binpan.manager.CandlesManager(symbol='btcusdt',
                                            tick_interval='15m',
                                            time_zone='Europe/Madrid',
                                            start_time='2021-10-31 01:00:00',
                                            end_time='2021-10-31 03:00:00')

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   binpan.rst
   market.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
