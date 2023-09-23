import pandas as pd
import psycopg2
from psycopg2 import sql
from typing import Tuple

from .exceptions import BinPanException
from .standards import postgresql_presentation_type_columns_dict, postgresql_response_cols_by_type
from .files import get_encoded_database_secrets
from .starters import AesCipher
from .logs import Logs

sql_logger = Logs(filename='./logs/sql.log', name='sql', info_level='INFO')
cipher_object = AesCipher()


def setup(symbol: str, tick_interval: str, postgres_klines: bool, postgres_atomic_trades: bool, postgres_agg_trades: bool) -> tuple:
    try:
        from secret import postgresql_host, postgresql_port, postgresql_user, postgresql_database
        enc_postgresql_password = get_encoded_database_secrets()
        connection, cursor = create_connection(user=postgresql_user,
                                               enc_password=enc_postgresql_password,
                                               host=postgresql_host,
                                               port=postgresql_port,
                                               database=postgresql_database)
        tables = get_valid_table_list(cursor=cursor)
        if postgres_klines:
            table = sanitize_table_name(f"{symbol.lower()}@kline_{tick_interval}")
            if not table in tables:
                raise BinPanException(f"BinPan Exception: Table {table} not found in database {postgresql_database}")
        if postgres_atomic_trades:
            table = sanitize_table_name(f"{symbol.lower()}@trade")
            if not table in tables:
                raise BinPanException(f"BinPan Exception: Table {table} not found in database {postgresql_database}")
        if postgres_agg_trades:
            table = sanitize_table_name(f"{symbol.lower()}@aggTrade")
            if not table in tables:
                raise BinPanException(f"BinPan Exception: Table {table} not found in database {postgresql_database}")
    except Exception as e:
        raise BinPanException(f"BinPan Exception: {e} \nVerify existence database credentials in secret.py (postgresql_host, "
                              "postgresql_port, postgresql_user, postgresql_password, postgresql_database)")
    return connection, cursor


# noinspection PyUnresolvedReferences
def create_connection(user: str,
                      enc_password: str,
                      host: str,
                      port: int,
                      database: str
                      ) -> Tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]:
    decoded_password = cipher_object.decrypt(enc_password)
    try:
        connection = psycopg2.connect(user=user,
                                      password=decoded_password,
                                      host=host,
                                      port=port,
                                      database=database)
        cursor = connection.cursor()
        sql_logger.debug(f"Conexión a PostgreSQL exitosa en ip {host} database {database}")
        return connection, cursor
    except (Exception, psycopg2.Error) as error:
        sql_logger.error(f"Error al conectar con PostgreSQL en ip {host} database {database}: {error}")

        return None, None


def data_type_from_table(table: str) -> str or None:
    """
    Gets data type from table names: "trade", "aggTrade", "depthX", "depth", "bookTicker" or "kline". "orderbook", "orderbook_value".

    :param str table: A stream name like "btcbusd_trade"
    :return str: Data type or none if not detected
    """

    if "_kline_" in table:
        return "kline"
    elif table.endswith("_aggTrade"):
        return "aggTrade"
    elif table.endswith("_trade"):
        return "trade"
    else:
        sql_logger.debug(f"get_valid_table_list: Tabla {table} no es un stream válido")


def get_valid_table_list(cursor) -> list:
    """
    Obtiene la lista de tablas en la base de datos.

    :param cursor: Un cursor de psycopg2 conectado a la base de datos.
    :return:
    """
    # noinspection SqlCurrentSchemaInspection
    query = "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema';"

    cursor.execute("BEGIN")
    cursor.execute(query)

    # Obtener y retornar la lista de tablas
    tables = [table[0] for table in cursor.fetchall()]

    cursor.execute("COMMIT")
    valid_tables = []
    for table in tables:
        my_type = data_type_from_table(table)
        if my_type:
            valid_tables.append(table)
    return sorted(valid_tables)


def sanitize_table_name(table_name: str) -> str:
    sanitized_name = table_name.replace("@", "_")

    # Añadir un prefijo si el nombre comienza con un número
    if sanitized_name[0].isdigit():
        sanitized_name = "t_" + sanitized_name

    return sanitized_name


# noinspection SqlCurrentSchemaInspection
def get_data_and_parse(cursor,
                       table: str,
                       symbol: str,
                       tick_interval: str,
                       time_zone: str,
                       start_time: int,
                       end_time: int,
                       data_type: str,
                       order_col: str,
                       ):
    # llama a la tabla de klines pedida en el intervalo de timestamps solicitado
    try:
        query = sql.SQL("SELECT * FROM {} WHERE EXTRACT(EPOCH FROM time) * 1000 >= {} AND EXTRACT(EPOCH FROM time) * 1000 <= {} ORDER BY time ASC;").format(
            sql.Identifier(table),
            sql.Literal(start_time),
            sql.Literal(end_time)
        )

        cursor.execute("BEGIN")
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.execute("COMMIT")
    except Exception as e:
        cursor.execute("ROLLBACK")
        sql_logger.error(f"Error obtaining data from table {table}: {e}")
        raise e
    # parsea los datos a un dataframe en base a las columnas de standards
    data_type_structure = postgresql_response_cols_by_type[data_type]
    data_dicts = [{data_type_structure[i]: l[i] for i in range(len(l))} for l in data]
    df = pd.DataFrame(data_dicts)

    if not "Timestamp" in df.columns and "Date" in df.columns:  # trades
        df['Timestamp'] = (df['Date'].astype('int64') // 10 ** 6).astype('int64')
    else:
        pass

    df['Date'] = df['Date'].dt.tz_convert(time_zone)
    df.set_index("Date", inplace=True)
    if order_col:
        df.sort_values(["Date", order_col], inplace=True)
    else:
        df.sort_values("Date", inplace=True)

    if data_type == "kline":
        df.index.name = f"{symbol.upper()} {tick_interval} {time_zone}"
    else:
        df.index.name = f"{symbol.upper()} {time_zone}"

    # forma de las columnas
    my_cols = postgresql_presentation_type_columns_dict[data_type]
    return df[my_cols]
