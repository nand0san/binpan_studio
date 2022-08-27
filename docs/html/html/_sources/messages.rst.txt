Messages Module
===============

This module can send messages through Telegram API for your bot and chat id.

To import this module:

.. code-block::

   from handlers import messages


.. automodule:: handlers.messages

Telegram Messages
-----------------

.. autofunction:: telegram_bot_send_text


Parsers
-------


.. autofunction:: telegram_parse_dict

.. autofunction:: telegram_parse_dataframe_markdown

.. autofunction:: telegram_parse_order_markdown


Utils
-----

.. autofunction:: send_balances
