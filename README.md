Welcome to BinPan's documentation!
==================================

BinPan is a Python wrapper for Binance API. Useful creating objects with many capabilities in data analysis.

BinPan can show plots easily and fetch API requests into the same object. It can also obtain some technical indicators.

The target of this module is to have a fast tool for collecting and handling data from the Binance API easily.

It is intended to be useful in Jupyter Notebooks or even the python console, but it can be used in
many other ways.

BinPan manages symbol objects that can do:

- get candles with time zone and indexing options.
- get trades.
- calculate technical indicators.
- plot candles, histograms, indicators, etc in a very simple and beautiful way.
- check applied fees.

An example of a plot for candles and indicators:

![](https://raw.githubusercontent.com/nand0san/binpan_studio/main/docs/images/candles.png)


Documentation
-------------
Take a look to the basic **tutorial**. Find it in the Jupyter Notebook file **tutorial.ipynb**

Also, can be found Sphinx documentation at: 

https://nand0san.github.io/binpan_studio/

Hope you find it useful breaking the market!!!

Python Version and Google Colab
-------------------------------

Please install in google colab with: `!pip install binpan`, then push restart runtime button and import library as usual.

Google colab ready: `python 3.7.9`

GitHub repo
-----------

https://github.com/nand0san/binpan_studio


Installation
------------
Pypi repository: https://pypi.org/project/binpan/

```
   pip install binpan
```

Usage
-----

Importing just like this:

```
    import binpan

    btcusdt = binpan.Symbol(symbol='btcusdt',
                            tick_interval='15m',
                            time_zone='Europe/Madrid',
                            start_time='2021-10-31 01:00:00',
                            end_time='2021-10-31 03:00:00')
                            
    btcusdt.sma(21)
    
    btcusdt.plot()
    
```

### Greetings

Thank you for the pandas_ta people for that great library.