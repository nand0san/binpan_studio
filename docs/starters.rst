Secrets Module
==============

Credential management delegated to ``panzer``.

BinPan implements no encryption of its own. All credentials (Binance API keys,
PostgreSQL/binbase passwords, Telegram tokens, Redis configs, etc.) are handled
by ``panzer``'s ``CredentialManager``, which stores them in ``~/.panzer_creds``
(sensitive values encrypted) and prompts the user the first time they are missing.

.. automodule:: binpan.core.secrets
   :members:
   :undoc-members:
   :show-inheritance:
