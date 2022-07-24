from .starters import AesCipher
import requests
from .logs import Logs
from .time_helper import convert_milliseconds_to_str

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
                           disable_notification=False,
                           alt_enc_bot_id: str = None,
                           alt_chat_id: str = None) -> dict:
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
    :param str alt_enc_bot_id: Optional alternative bot api key.
    :param str alt_chat_id: Optional alternative chat id.
    :return: Telegrams API response.
    """

    if type(msg) == dict:
        bot_message = msg.copy()
    else:
        bot_message = msg

    if not alt_enc_bot_id:
        alt_enc_bot_id = encoded_telegram_bot_id
    if not alt_chat_id:
        alt_chat_id = encoded_chat_id

    bot_message = str(bot_message)
    alt_enc_bot_id = cipher_object.decrypt(bot_id)
    alt_chat_id = cipher_object.decrypt(chat_id)

    if disable_notification:
        send_text = f'https://api.telegram.org/bot{alt_enc_bot_id}/sendMessage?chat_id={alt_chat_id}&' \
                    f'parse_mode={parse_mode}={bot_message}&disable_notification=true'

    else:
        send_text = f'https://api.telegram.org/bot{alt_enc_bot_id}/sendMessage?chat_id={alt_chat_id}&' \
                    f'parse_mode={parse_mode}={bot_message} '
    response = requests.get(send_text)

    msg_logger.info("TELEGRAM " + " ".join(bot_message.replace('\n', ' ').split()))

    return response.json()


def telegram_parse_dict(msg_data: dict, timezone='UTC'):
    """
    Parses a dict and downcast types.

    :param dict msg_data: Dict to parse as message.
    :param str timezone: A time zone to parse more human readable any field containing "time" in the name.
    :return str: A markdown v1 parsed telegram string.
    """
    msg_dict = msg_data.copy()
    parsed_msg = ""
    for k, v in msg_dict.items():
        try:
            fv1 = float(v)
        except:
            fv1 = v
        try:
            fv2 = int(fv1)
            assert fv2 == fv1
        except:
            fv2 = fv1
        if 'pct' in k:
            fv2 = fv2*100
            row = f"*{k}* : `{fv2:.2f}`\n"
        elif type(fv2) == float:
            row = f"*{k}* : `{fv2:.8f}`\n"
        elif type(fv2) == int:
            row = f"*{k}* : `{fv2}`\n"
        else:
            if 'time' in k:
                date = convert_milliseconds_to_str(ms=fv2, timezoned=timezone)
                row = f"*{k}* : {date}\n"
            else:
                row = f"*{k}* : {fv2}\n"
        parsed_msg += row
    return parsed_msg
