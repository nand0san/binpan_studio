"""
Functions to initialize before starting anything.
"""
from base64 import b64encode, b64decode
import hashlib
from binascii import unhexlify
from os.path import expanduser
from cpuinfo import get_cpu_info
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import sys
import os
import importlib


def import_secret_module():
    """
    Imports the 'secret.py' module from anywhere in the project directory hierarchy.

    The function searches for the 'secret.py' module in the current directory and all upper directories until it finds it
    or reaches the root of the file system.

    :return: The 'secret' module if found, None otherwise.
    :rtype: Module or None
    """
    current_dir = os.path.abspath(os.curdir)

    while True:
        try:
            secret_module = importlib.import_module('secret')
            # regex = r"(?<=\\Users\\)([^\\]+)"
            # obfuscated_path = re.sub(regex, "XXXX", current_dir)
            # print("SECRET module found: ", obfuscated_path)
            return secret_module
        except ModuleNotFoundError:
            # Si no se encuentra el módulo, sube un nivel en el directorio
            parent_dir = os.path.dirname(current_dir)

            # Si ya estamos en la raíz del sistema de archivos, detener la búsqueda
            if parent_dir == current_dir:
                raise ModuleNotFoundError(
                    "BinPan: 'secret.py' no encontrado en ningún directorio padre. "
                    "Necesario para operaciones con claves API. "
                    "Ejecuta las funciones de setup de credenciales primero."
                )

            current_dir = parent_dir
            sys.path.insert(0, current_dir)


class AesCipher(object):
    """
    Cyphering object.

    Initialization function. Generates a key and an initialization vector based on the CPU info and the user home.

    """

    def __init__(self):
        __seed = bytes(expanduser("~") + get_cpu_info()['brand_raw'], "utf8")
        self.__iv = hashlib.md5(__seed).hexdigest()
        self.__key = hashlib.md5(__seed[::-1]).hexdigest()

    def encrypt(self, msg: str) -> str:
        """
        Encrypting function. Encrypts a message with AES-128-CBC.

        :param str msg: Any message to encrypt.
        :return str: A bytes base64 string.
        """
        msg_padded = pad(msg.encode(), AES.block_size)
        cipher = AES.new(unhexlify(self.__key), AES.MODE_CBC, unhexlify(self.__iv))
        cipher_text = cipher.encrypt(msg_padded)
        return b64encode(cipher_text).decode('utf-8')

    def decrypt(self, msg_encrypted: str) -> str:
        """
        Decrypt function. Decrypts a message with AES-128-CBC.

        :param str msg_encrypted: A bytes base64 string.
        :return str: Plain text.
        """
        decipher = AES.new(unhexlify(self.__key), AES.MODE_CBC, unhexlify(self.__iv))
        plaintext = unpad(decipher.decrypt(b64decode(msg_encrypted)), AES.block_size).decode('utf-8')
        return plaintext


