Welcome to BinPan's documentation!
==================================

BinPan is a Python wrapper for Binance API, creating objects with many capabilities to analyze data.

BinPan can make plots easily and grab API requests in the same object. It can also obtain some technical indicators.

The objective of this module is to have a fast tool to collect and handle data from the Binance API
in an agile way.

It is intended to be useful in Jupyter Notebooks or even the python console, but it can be used in
many other ways.

BinPan manages symbol objects that can:

- get candles with time zone and indexing options.
- get trades.
- calculate technical indicators.
- plot candles, histograms, indicators, etc in a very simple and beautiful way.
- check applied fees.

An example of a plot for candles and indicators:

![Alt text](./doc/images/candles.png?raw=true "Candles example")

Documentation is in html format in the file:

`./doc/_build/html/index.html`

Hope you find it useful breaking the market!!!


Installation
------------

```
   pip install binpan
```

Usage
-----

Importing just like this:

```
    import binpan

    btcusdt = binpan.manager.CandlesManager(symbol='btcusdt',
                                            tick_interval='15m',
                                            time_zone='Europe/Madrid',
                                            start_time='2021-10-31 01:00:00',
                                            end_time='2021-10-31 03:00:00')
    btcusdt.sma(21)
    btcusdt.plot()
```
The repository contains a Jupyter Notebook with many examples.