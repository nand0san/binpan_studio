"""
Secret management delegated to panzer.

BinPan does not implement its own encryption. All credentials (Binance API
keys, PostgreSQL/binbase passwords, Redis configs, etc.) are handled by
:class:`panzer.credentials.CredentialManager`, which stores them in
``~/.panzer_creds`` with three layers: in-memory cache, on-disk file (sensitive
values encrypted with AES-128-CBC) and an interactive prompt when a credential
is missing.

panzer auto-detects sensitive names by the markers ``secret``, ``api_key``,
``password`` and ``_id``. Choose credential names accordingly so that
sensitive values are encrypted and transparently decrypted on read:

- ``postgresql_password`` / ``binbase_password`` -> encrypted (``password``)
- ``postgresql_host`` / ``postgresql_user`` / ... -> stored as plain text
- ``redis_conf`` / ``sentinel_data`` -> JSON, plain text (see ``*_json_secret``)
"""
from __future__ import annotations

import json

from panzer.credentials import CredentialManager

_manager: CredentialManager | None = None


def _cm() -> CredentialManager:
    """Lazy singleton of panzer's CredentialManager (avoids CPU info at import)."""
    global _manager
    if _manager is None:
        _manager = CredentialManager()
    return _manager


def get_secret(name: str, *, decrypt: bool = True) -> str:
    """
    Get a credential as plain text. Resolves memory -> disk -> prompt.

    :param str name: Credential name (e.g. ``"postgresql_password"``).
    :param bool decrypt: If True, decrypt sensitive values before returning.
    :return str: The credential value in plain text.
    """
    return _cm().get(name, decrypt=decrypt)


def set_secret(name: str, value: str, *, sensitive: bool | None = None, overwrite: bool = True) -> None:
    """
    Store a credential in panzer (encrypted on disk if sensitive).

    :param str name: Credential name.
    :param str value: Plain text value.
    :param sensitive: Force encryption (True/False) or auto-detect by name (None).
    :param bool overwrite: Overwrite if it already exists on disk.
    """
    _cm().add(name, value, sensitive=sensitive, overwrite=overwrite)


def get_json_secret(name: str) -> dict:
    """
    Get a credential serialized as JSON (e.g. Redis / Sentinel configs).

    :param str name: Credential name.
    :return dict: The deserialized value.
    """
    return json.loads(get_secret(name, decrypt=True))


def set_json_secret(name: str, value: dict, *, overwrite: bool = True) -> None:
    """
    Store a dict credential as a JSON string in panzer.

    :param str name: Credential name.
    :param dict value: The dictionary to serialize and store.
    :param bool overwrite: Overwrite if it already exists on disk.
    """
    set_secret(name, json.dumps(value), sensitive=False, overwrite=overwrite)
