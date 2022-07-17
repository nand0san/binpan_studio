from .starters import AesCipher
import requests
from .logs import Logs

msg_logger = Logs(filename='./logs/msg_logger.log', name='msg_logger', info_level='INFO')

cipher_object = AesCipher()

try:
    from secret import encoded_chat_id, encoded_telegram_bot_id

    chat_id = encoded_chat_id
    bot_id = encoded_telegram_bot_id
except Exception as exc:
    msg = """
Not found telegram bot key or chat key for the telegram message module.

Example adding bot key and chat id:
    
    from binpan import handlers

    # from @BotFather, get the bot api key
    handlers.files.add_any_key(key="xxxxxxxxxx", key_name="encoded_telegram_bot_id")

    # write to your bot and then get your chat id from https://api.telegram.org/bot<YourBOTToken>/getUpdates
    handlers.files.add_any_key(key="xxxxxxxxxx", key_name="encoded_chat_id")
"""
    msg_logger.warning(msg)
    encoded_telegram_bot_id = ''
    encoded_chat_id = ''


def telegram_bot_send_text(msg: dict or str,
                           parse_mode='Markdown&text',
                           disable_notification=False) -> dict:
    """
    Sends a telegram message.

    It takes bot key and chat id from secret encrypted file. It is required to create previously the bot and chat keys with the
    variable names *encoded_chat_id*, *encoded_telegram_bot_id*.

    Example adding bot key and chat id:

    .. code-block:: python

        from binpan import handlers

        # from @BotFather, get the bot api key
        handlers.files.add_any_key(key="xxxxxxxxxx", key_name="encoded_telegram_bot_id")

        # write to your bot and then get your chat id from https://api.telegram.org/bot<YourBOTToken>/getUpdates
        handlers.files.add_any_key(key="xxxxxxxxxx", key_name="encoded_chat_id")


    :param str msg: A message
    :param str parse_mode: Parsing telegram message. Default is 'Markdown&text'.
    :param disable_notification: Avoids getting notified by the telegram message. Only working in group chats.
        https://stackoverflow.com/questions/64360221/disable-notification-in-python-telegram-bot
    :return: Telegrams API response.
    """

    if type(msg) == dict:
        bot_message = msg.copy()
    else:
        bot_message = msg

    bot_message = str(bot_message)
    encoded_telegram_bot_id = cipher_object.decrypt(bot_id)
    encoded_chat_id = cipher_object.decrypt(chat_id)

    if disable_notification:
        send_text = f'https://api.telegram.org/bot{encoded_telegram_bot_id}/sendMessage?chat_id={encoded_chat_id}&' \
                    f'parse_mode={parse_mode}={bot_message}&disable_notification=true'

    else:
        send_text = f'https://api.telegram.org/bot{encoded_telegram_bot_id}/sendMessage?chat_id={encoded_chat_id}&' \
                    f'parse_mode={parse_mode}={bot_message} '
    response = requests.get(send_text)

    msg_logger.info("TELEGRAM " + " ".join(bot_message.replace('\n', ' ').split()))

    return response.json()
