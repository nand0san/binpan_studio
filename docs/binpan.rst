BinPan Module
=============

Can be imported this way:

.. code-block::

   import binpan

.. automodule:: binpan


Symbol Class
------------

This class can manage klines, trades, and many more data.

Also includes technical indicators, makes plots, and gets some exchange data.

There are tutorials in a Jupyter Notebook file at https://github.com/nand0san/binpan_studio

.. autoclass:: Symbol
   :members:


Exchange Class
--------------

This class is oriented towards managing data from exchanges, like Symbols status, coins networks, etc.

There is a tutorial in a Jupyter Notebook file at https://github.com/nand0san/binpan_studio

.. autoclass:: Exchange
   :members:


Wallet Class
------------

This class can show wallet data for you. Free assets, locked assets, or wallet snapshots for
performance analysis.

.. autoclass:: Wallet
   :members:

Jupyter Import Problems Troubleshooting
---------------------------------------

When working with Jupyter, you may encounter import errors while trying to import packages such as BinPan.
These errors can be caused by various reasons such as package installation order, virtual environment issues, etc.
To resolve such errors, you can try installing the required modules directly to the Jupyter Notebook kernel by
following the steps below:

First, import the sys module in your Jupyter notebook.

Next, install the required packages using the following command:

.. code-block::

    import sys

    !{sys.executable} -m pip install <package_name>

Replace <package_name> with the name of the package that you want to install.

By following these steps, you can install the required packages directly to the Jupyter Notebook kernel and resolve any import
errors that you may encounter. In addition, it is recommended to ensure that the virtual environment used by Jupyter is
configured correctly to avoid any conflicts with package installations.