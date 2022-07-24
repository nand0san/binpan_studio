Files Module
============

This module can manage files and API keys in an encrypted API keys and secrets file called *secret.py*.

Can be imported this way:

.. code-block::

   from handlers import files


.. automodule:: handlers.files

Files and Folders
-----------------

.. autofunction:: create_dir

.. autofunction:: save_dataframe_to_csv

.. autofunction:: append_dataframe_to_csv

.. autofunction:: find_csvs_in_path

.. autofunction:: move_old_csvs

.. autofunction:: read_csv_to_dataframe

.. autofunction:: read_file

.. autofunction:: save_file

Secrets Archive
---------------

To add encrypted data to secret.py file.


.. autofunction:: add_api_key

.. autofunction:: add_api_secret

.. autofunction:: add_any_key
