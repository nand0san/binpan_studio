Requests Module
=========================

This module manage requests and weight for API interactions.

To import this module:

.. code-block::

   from handlers import quest

.. automodule:: handlers.quest


Tools
------------

.. autofunction:: add_header_for_endpoint

.. autofunction:: update_weights

.. autofunction:: check_weight

.. autofunction:: get_server_time


Requests
--------

.. autofunction:: convert_response_type

.. autofunction:: get_response

.. autofunction:: post_response

.. autofunction:: delete_response

.. autofunction:: handle_api_response

.. autofunction:: hashed_signature

.. autofunction:: sign_request

.. autofunction:: get_signed_request

.. autofunction:: get_semi_signed_request

.. autofunction:: post_signed_request

.. autofunction:: delete_signed_request


Request Shortcuts
-----------------

.. autofunction:: api_raw_get

.. autofunction:: api_raw_signed_get

.. autofunction:: api_raw_signed_post



