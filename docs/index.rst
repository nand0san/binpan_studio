Welcome to BinPan's documentation!
==================================

BinPan is a Python wrapper for Binance API. Useful creating objects with many capabilities in data analysis.

BinPan can show plots easily and fetch API requests into the same object. It can also obtain some technical indicators.

The target of this module is to have a fast tool for collecting and handling data from the Binance API easily.

It is intended to be useful in Jupyter Notebooks or even the python console, but it can be used in many other ways.

BinPan manages symbol objects that can do:

- get candles with time zone and indexing options.
- get trades.
- calculate technical indicators.
- plot candles, histograms, indicators, etc in a very simple and beautiful way.
- check applied fees.

An example of a plot for candles and indicators:

.. image:: images/candles.png
   :width: 1000
   :alt: Candles with some indicators

.. note::

    BinPan contains no Binance order method, withdraw method or any dangerous command.

    If you decide to add API keys for using some account methods, BinPan will encrypt it in a file, and in memory,
    but it is better not enabling trading capability on the Binance API key configuration, just for your own peace of mind.

    Be careful out there!


Hope you find it useful breaking the market!!!


Documentation
-------------

Full documentation can be found at:

https://nand0san.github.io/binpan_studio/

Take a look to the basic **tutorial**. Find it in the Jupyter Notebook file **tutorial.ipynb**

There are many more tutorials in Jupyter Notebooks at https://github.com/nand0san/binpan_studio

Hope you find it useful breaking the market!!!

Google Colab
-------------------------------

Google Colab is not available for the Binance API. Maybe Colab's IPs are restricted in the binance API.

.. code-block::

    BinanceAPIException: APIError(code=0): Service unavailable from a restricted location according to 'b. Eligibility' in https://www.binance.com/en/terms. Please contact customer service if you believe you received this message in error.



GitHub repo
-----------
https://github.com/nand0san/binpan_studio


Installation
------------

Pypi repository: https://pypi.org/project/binpan/

.. code-block:: bash

   pip install binpan

Usage
-----

There is a tutorial in a Jupyter Notebook file in  the github repo.

https://github.com/nand0san/binpan_studio/blob/main/basic%20tutorial.ipynb

Importing just like this:

.. code-block:: python

    import binpan

    btcusdt = binpan.Symbol(symbol='btcusdt',
                            tick_interval='15m',
                            time_zone='Europe/Madrid',
                            end_time='2021-10-31 03:00:00')
    btcusdt.sma(21)
    btcusdt.plot()


.. image:: images/index_example.png
   :width: 1000


Greetings
---------

Thanks to the **pandas_ta** people for that great library.

Hope you find it useful breaking the market!!!


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   binpan.rst
   aggregations.rst
   exceptions.rst
   exchange.rst
   files.rst
   indicators.rst
   market.rst
   messages.rst
   plotting.rst
   quest.rst
   redis_fetch.rst
   stat_tests.rst
   strategies.rst
   tags.rst
   time_helper.rst
   wallet.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
