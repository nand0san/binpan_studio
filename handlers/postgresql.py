import pandas as pd
import psycopg2
from psycopg2 import sql
from typing import Tuple, List, Dict, Optional
from time import sleep

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import tqdm

from .exceptions import BinPanException
from .standards import *
from .files import get_encoded_database_secrets
from .starters import AesCipher
from .logs import Logs
from .market import convert_to_numeric
from .time_helper import convert_milliseconds_to_str

sql_logger = Logs(filename='./logs/sql.log', name='sql', info_level='INFO')
cipher_object = AesCipher()


def setup(symbol: str,
          tick_interval: str,
          postgresql_host: str,
          postgresql_user: str,
          postgresql_database: str,
          postgres_klines: bool,
          postgres_atomic_trades: bool,
          postgres_agg_trades: bool,
          postgresql_port: int = 5432) -> tuple:
    """
    Setups the connection to the PostgreSQL database.

    :param symbol: A symbol like "BTCUSDT"
    :param tick_interval: A tick interval like "1m"
    :param postgresql_host: A host name or ip address
    :param postgresql_user: A user name
    :param postgresql_database: A database name
    :param postgres_klines: A boolean to indicate if klines are requested
    :param postgres_atomic_trades: A boolean to indicate if atomic trades are requested
    :param postgres_agg_trades: A boolean to indicate if aggregate trades are requested
    :param postgresql_port: A port number
    :return: Connection and cursor to the database

    """
    try:
        # from secret import postgresql_host, postgresql_port, postgresql_user, postgresql_database
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
            table = sanitize_table_name(f"{symbol.lower()}@aggtrade")
            if not table in tables:
                raise BinPanException(f"BinPan Exception: Table {table} not found in database {postgresql_database}")
    except Exception as e:
        raise BinPanException(f"BinPan Exception: {e} \nVerify existence on TABLE or DATABASE CREDENTIALS in secret.py "
                              f"(postgresql_host, postgresql_port, postgresql_user, postgresql_password, postgresql_database)")
    return connection, cursor


# noinspection SqlCurrentSchemaInspection
def get_data_and_parse(cursor,
                       table: str,
                       symbol: str,
                       tick_interval: str,
                       time_zone: str,
                       start_time: int,
                       end_time: int,
                       data_type: str,
                       ):
    """
    Gets data from a table in the database and parses it to a dataframe.

    :param cursor: Database cursor
    :param table: Table name
    :param symbol: Symbol like "BTCUSDT"
    :param tick_interval: Tick interval like "1m"
    :param time_zone: Time zone like "Europe/Madrid"
    :param start_time: Timestamp in milliseconds
    :param end_time: Timestamp in milliseconds
    :param data_type: Data type like "kline", "trade", "aggTrade". Types are Binance websocket stream name types.
    :return: A dataframe with the data and pertinent columns.
    """
    # llama a la tabla de klines pedida en el intervalo de timestamps solicitado
    start_time_string = convert_milliseconds_to_str(start_time, timezoned=time_zone)
    end_time_string = convert_milliseconds_to_str(end_time, timezoned=time_zone)
    sql_logger.info(f"Getting data from table {table} from {start_time_string} to {end_time_string}")

    try:
        query = sql.SQL("SELECT * FROM {} WHERE EXTRACT(EPOCH FROM time) * 1000 >= {} "
                        "AND EXTRACT(EPOCH FROM time) * 1000 <= {} ORDER BY "
                        "time ASC;").format(sql.Identifier(table),
                                            sql.Literal(start_time),
                                            sql.Literal(end_time))
        cursor.execute("BEGIN")
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.execute("COMMIT")
    except Exception as e:
        cursor.execute("ROLLBACK")
        sql_logger.error(f"Error obtaining data from table {table}: {e}")
        raise e

    # parsea los datos a un dataframe en base a las columnas de standards
    data_type_structure = get_column_names(cursor=cursor, table_name=table, own_transaction=True)
    data_dicts = [{data_type_structure[i]: l[i] for i in range(len(l))} for l in data]
    df = pd.DataFrame(data_dicts)
    df.rename(columns=postgresql2binpan_renamer_dict[data_type], inplace=True)
    alt_order = None
    if data_type == "trade":
        if trade_date_col in df.columns:
            df[trade_time_col] = (df[trade_date_col].astype('int64') // 10 ** 6).astype('int64')
            df[trade_date_col] = df[trade_date_col].dt.tz_convert(time_zone)
        else:  # si no trae la columna de date, es q viene de timescale y timestamp esta en formato datetime
            df[trade_date_col] = df[trade_time_col]
            # Convertir a la zona horaria deseada
            df[trade_date_col] = df[trade_date_col].dt.tz_convert(time_zone)
            # timestamp a milisegundos
            df[trade_time_col] = (df[trade_time_col].astype('int64') // 10 ** 6).astype('int64')

        date_col = trade_date_col
        alt_order = trade_trade_id_col
    elif data_type == "kline":
        df[kline_open_time_col] = df[kline_open_time_col].dt.tz_convert(time_zone)
        df[kline_open_timestamp_col] = (df[kline_open_time_col].astype('int64') // 10 ** 6).astype('int64')

        df[kline_close_time_col] = df[kline_close_timestamp_col]

        # esto falla cuando se llama a timescale
        df[kline_close_time_col] = pd.to_datetime(df[kline_close_timestamp_col], unit='ms')
        df[kline_close_time_col] = df[kline_close_time_col].dt.tz_localize('UTC')  # Cambia 'UTC' si es necesario
        df[kline_close_time_col] = df[kline_close_time_col].dt.tz_convert(time_zone)

        date_col = kline_open_time_col

    elif data_type == "aggTrade" or data_type == "aggtrade":
        df[agg_time_col] = (df[agg_time_col].astype('int64') // 10 ** 6).astype('int64')
        if agg_date_col in df.columns:
            df[agg_date_col] = df[agg_date_col].dt.tz_convert(time_zone)
        else:
            # Convertir milisegundos a datetime
            df[agg_date_col] = pd.to_datetime(df[agg_time_col], unit='ms')
            df[agg_date_col] = df[agg_date_col].dt.tz_localize('UTC')  # Cambia 'UTC' si es necesario
            # Convertir a la zona horaria deseada
            df[agg_date_col] = df[agg_date_col].dt.tz_convert(time_zone)

        date_col = agg_date_col
        alt_order = agg_trade_id_col
    else:
        raise Exception(f"get_data_and_parse: BinPan Exception: Table {table} not recognized as a valid table")

    df.set_index(date_col, inplace=True, drop=False)

    if alt_order:
        df.index.name = None
        df.sort_values([date_col, alt_order], inplace=True)
    else:
        df.sort_index(inplace=True)

    if data_type == "kline":
        df.index.name = f"{symbol.upper()} {tick_interval} {time_zone}"
    else:
        df.index.name = f"{symbol.upper()} {time_zone}"
    df = convert_to_numeric(data=df)
    ordered_existing_cols = [col for col in postgresql_presentation_type_columns_dict[data_type] if col in df.columns]
    not_expected_cols = [col for col in df.columns if col not in postgresql_presentation_type_columns_dict[data_type]]
    return df[ordered_existing_cols + not_expected_cols]


# A PARTIR DE AQUÍ ES IGUAL QUE EN postgresql_utils de TIMESCALE

###############
# connections #
###############


# noinspection PyUnresolvedReferences
def create_connection(user: str,
                      enc_password: str,
                      host: str,
                      port: int,
                      database: str,
                      timeout: int = 10
                      ) -> Tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]:
    """
    Crea una conexión a la base de datos PostgreSQL.

    :param user: Nombre de usuario
    :param enc_password: Contraseña cifrada.
    :param host: Nombre del host o dirección IP
    :param port: Número de puerto
    :param database: Nombre de la base de datos
    :param timeout: Tiempo de espera en segundos
    :return: Devuelve una conexión y un cursor a la base de datos
    """
    decoded_password = cipher_object.decrypt(enc_password)
    try:
        connection = psycopg2.connect(user=user,
                                      password=decoded_password,
                                      host=host,
                                      port=port,
                                      database=database,
                                      connect_timeout=timeout)
        cursor = connection.cursor()
        sql_logger.debug(f"Conexión exitosa a PostgreSQL en ip {host} database {database}")
        return connection, cursor
    except (Exception, psycopg2.Error) as error:
        if "database" in str(error).lower() and "does not exist" in str(error).lower():
            # Conectarse a la base de datos predeterminada para crear la nueva base de datos
            temp_conn = psycopg2.connect(user=user,
                                         password=decoded_password,
                                         host=host,
                                         port=port,
                                         database="postgres",  # Base de datos predeterminada
                                         connect_timeout=timeout)
            temp_cursor = temp_conn.cursor()
            create_database(temp_cursor, database)
            temp_conn.commit()
            temp_cursor.close()
            temp_conn.close()
            # Intentar la conexión de nuevo
            return create_connection(user, enc_password, host, port, database, timeout)
        else:
            msg = f"Error al conectar a PostgreSQL {host} database {database}: {error}"
            sql_logger.error(msg)
            raise BinPanException(msg)


def is_cursor_alive(cursor):
    """
    Verifica si un cursor de PostgreSQL está activo y en buen estado.

    Parámetros:
    - cursor: Cursor de la conexión a la base de datos PostgreSQL.

    Retorna:
    - True si el cursor está en buen estado, False en caso contrario.
    """
    try:
        cursor.execute("SELECT 1;")
        return True
    except Exception as e:
        print(f"Cursor is not alive: {e}")
        return False


def close_connection(connection, cursor):
    """
    It closes the connection to the PostgreSQL database.

    :param connection: A psycopg2 connection to the database.
    :param cursor: A psycopg2 cursor to the database.
    :return: It returns nothing.
    """
    if connection:
        cursor.close()
        connection.close()
        sql_logger.debug(f"Connection to PostgreSQL closed.")


######################
# funciones atómicas #
######################

def create_database(cursor, db_name: str):
    """
    Crea una nueva base de datos en PostgreSQL.

    :param cursor: Cursor de psycopg2 a la base de datos.
    :param db_name: Nombre de la base de datos a crear.
    """
    cursor.execute(f"CREATE DATABASE {db_name};")


def list_tables_with_suffix(cursor, suffix="") -> list:
    """
    Retorna una lista con los nombres de las tablas que terminan con el sufijo especificado.

    :param cursor: Un cursor de psycopg2 conectado a la base de datos.
    :param suffix: Opcional. Sufijo de las tablas a buscar. Ejemplo: "_trade".
    :return: Una lista con los nombres de las tablas que terminan con el sufijo especificado.
    """
    if suffix:
        suffix = suffix.lower()
    if suffix:
        query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name LIKE '%{suffix}'
          AND table_schema = 'public';  -- Opcional, si quieres filtrar por esquema
        """
    else:
        query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public';  -- Opcional, si quieres filtrar por esquema
        """
    cursor.execute(query)
    return [row[0] for row in cursor.fetchall()]


def count_rows_in_tables(cursor, table_names: List[str], approximated: bool = True) -> Dict[str, int]:
    counts = {}
    for table in table_names:
        if approximated:
            cursor.execute(f"SELECT approximate_row_count('{table}')")
        else:
            cursor.execute(f"SELECT COUNT(*) FROM {table};")
        count = cursor.fetchone()[0]
        counts[table] = count
    return counts


def is_hypertable(cursor, table_name):
    """
    Si la tabla es una hypertable, retorna True. Si no, retorna False.
    :param cursor:
    :param table_name:
    :return:
    """
    query = f"""
    SELECT * 
    FROM timescaledb_information.hypertables 
    WHERE hypertable_name = '{table_name}';
    """
    cursor.execute(query)
    return cursor.fetchone() is not None


def get_column_names(cursor,
                     table_name: str,
                     own_transaction: bool) -> list:
    """
    Returns a list with the names of the columns in a table.

    :param cursor: A psycopg2 cursor connected to the database.
    :param table_name: Name of the table.
    :param own_transaction: If True, the function will create its own transaction. If False, the function will use the
     transaction of the cursor.
    :return:
    """
    query = f"""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = '{table_name}';
    """
    try:
        if own_transaction:
            cursor.execute("BEGIN")
        cursor.execute(query)
        columns = [row[0] for row in cursor.fetchall()]
        if own_transaction:
            cursor.execute("COMMIT")
    except Exception as _:
        if own_transaction:
            cursor.execute("ROLLBACK")
        columns = []
        assert is_cursor_alive(cursor), "Cursor no está vivo"
    return columns


def fetch_hypertable(cursor, table: str, startime: int = None, endtime: int = None) -> List[tuple]:
    """
    Returns a list with integrity data, that is, trades or klines. Obtained from the table.

    :param cursor: A psycopg2 cursor connected to the database.
    :param table: Name of the table.
    :param startime: A timestamp in milliseconds. If not specified, the first record will be used.
    :param endtime: A timestamp in milliseconds. If not specified, it will fetch up to the last record.
    :return: It returns a list of tuples with the data retrieved.
    """
    if startime:
        assert type(startime) is int, f"startime debe ser un entero: {startime}"
    if endtime:
        assert type(endtime) is int, f"endtime debe ser un entero: {endtime}"

    if not startime and not endtime:
        query = f"SELECT * FROM {table}"
    elif startime and not endtime:
        query = f"SELECT * FROM {table} WHERE time >= to_timestamp({startime} / 1000.0)"
    elif not startime and endtime:
        query = f"SELECT * FROM {table} WHERE time <= to_timestamp({endtime} / 1000.0"
    else:
        query = f"SELECT * FROM {table} WHERE time >= to_timestamp({startime} / 1000.0) AND time <= to_timestamp({endtime} / 1000.0)"
    try:
        cursor.execute("BEGIN")
        cursor.execute(query)
        ret = cursor.fetchall()
        cursor.execute("COMMIT")
    except Exception as e:
        sql_logger.debug(f"Error al obtener datos de la tabla: {e}")
        cursor.execute("ROLLBACK")
        ret = []
        assert is_cursor_alive(cursor), "Cursor no está vivo"

    return ret


def fetch_hypertable_selective(cursor,
                               table: str,
                               columns: Optional[List[str]] = None,
                               startime: int = None,
                               endtime: int = None) -> List[tuple]:
    """
    Returns a list with the specified data from a table.

    :param cursor: A psycopg2 cursor connected to the database.
    :param table: Name of the table.
    :param columns: List of columns to select.
    :param startime: A timestamp in milliseconds. If not specified, the first record will be used.
    :param endtime: A timestamp in milliseconds. If not specified, it will fetch up to the last record.
    :return: List of tuples with the recovered data.
    """

    if columns:
        assert all(isinstance(col, str) for col in columns), "Todas las columnas deben ser cadenas."
        select_clause = ', '.join(columns)
    else:
        select_clause = '*'

    if startime:
        assert type(startime) is int, f"startime debe ser un entero: {startime}"
    if endtime:
        assert type(endtime) is int, f"endtime debe ser un entero: {endtime}"

    query = f"SELECT {select_clause} FROM {table}"

    if startime and endtime:
        query += f" WHERE time >= to_timestamp({startime} / 1000.0) AND time <= to_timestamp({endtime} / 1000.0)"
    elif startime:
        query += f" WHERE time >= to_timestamp({startime} / 1000.0)"
    elif endtime:
        query += f" WHERE time <= to_timestamp({endtime} / 1000.0)"

    try:
        cursor.execute("BEGIN")
        cursor.execute(query)
        ret = cursor.fetchall()
        cursor.execute("COMMIT")
    except Exception as e:
        sql_logger.debug(f"Error al obtener datos de la tabla: {e}")
        cursor.execute("ROLLBACK")
        ret = []
        assert is_cursor_alive(cursor), "Cursor no está vivo"

    return ret


def delete_record(cursor, table_name: str, field_name: str, value: any, is_timestamp: bool = False, own_transaction=True):
    """
    Delete records from a hypertable in TimescaleDB where a specific field matches a specific value.

    :param cursor: Cursor from psycopg2 to the database.
    :param table_name: Name of the hypertable.
    :param field_name: Name of the field to compare.
    :param value: Value to look for to delete.
    :param is_timestamp: If True, converts the millisecond value to TIMESTAMPZ.
    :param own_transaction: If True, the function will create its own transaction. If False, the function will use the
    """
    if own_transaction:
        cursor.execute("BEGIN")
    if is_timestamp:
        query = sql.SQL("DELETE FROM {} WHERE {} = to_timestamp(%s / 1000.0);").format(
            sql.Identifier(table_name),
            sql.Identifier(field_name))
    else:
        query = sql.SQL("DELETE FROM {} WHERE {} = %s;").format(
            sql.Identifier(table_name),
            sql.Identifier(field_name))

    # Ejecutar la consulta
    cursor.execute(query, [value])
    if own_transaction:
        cursor.execute("COMMIT")


def delete_table(cursor, table_name, schema='public'):
    """
    Delete a table from the database to which the cursor is connected. Also removes the hypertable if it exists.

    :param cursor: A psycopg2 cursor connected to the database.
    :param table_name: Name of the table to delete.
    :param schema: Table schema.
    :return: Returns True if the table was deleted successfully, False otherwise.
    """
    try:
        drop_query = f"DROP TABLE IF EXISTS {schema}.{table_name};"
        cursor.execute(drop_query)
        return True
    except Exception as e:
        print(f"Error al eliminar la tabla: {e}")
        return False


def delete_bulk_tables(cursor, tables: list, batch=20):
    """
    It deletes a list of tables from the database to which the cursor is connected.

    :param cursor: A psycopg2 cursor connected to the database.
    :param tables: A list of tables to delete.
    :param batch: A batch size. Default is 20. Each batch will be committed.
    :return: It returns nothing.
    """
    deleted = []
    counter = 0

    try:
        cursor.execute("BEGIN;")
        pbar = tqdm(tables)
        for table in pbar:
            pbar.set_description(f"Processing {table}")
            if not is_cursor_alive(cursor):
                print("Cursor no está vivo. Reintentando...")
                cursor.execute("ROLLBACK;")
                sleep(5)  # Esperar 5 segundos
                cursor.execute("BEGIN;")

            drop_query = f"DROP TABLE IF EXISTS {table};"
            cursor.execute(drop_query)
            deleted.append(table)

            counter += 1
            if counter % batch == 0:
                cursor.execute("COMMIT;")
                sql_logger.info(f"Eliminadas {counter} tablas. Waiting 5 seconds...")
                sleep(5)
                cursor.execute("BEGIN;")

        cursor.execute("COMMIT;")
    except Exception as e:
        print(f"Error al eliminar tablas: {e}")
        cursor.execute("ROLLBACK;")


########################
# funciones alto nivel #
########################

def sanitize_table_name(table_name: str) -> str:
    """
    Sanitizes a table name. It replaces @ with _ and adds a prefix if the name starts with a number.

    :param table_name: A table name.
    :return: A sanitized table name.
    """
    sanitized_name = table_name.replace("@", "_")

    # Añadir un prefijo si el nombre comienza con un número
    if sanitized_name[0].isdigit():
        sanitized_name = "t_" + sanitized_name
    if table_name.endswith("aggTrade"):
        sanitized_name = sanitized_name.replace("aggTrade", "aggtrade")
    return sanitized_name.lower()


def get_valid_table_list(cursor) -> list:
    """
    Gets the list of tables in the database.

    :param cursor: A psycopg2 cursor connected to the database.
    :return:
    """
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


def infer_sql_type(value: any, key: str, time_col: str = "time") -> str:
    """
    Infers the SQL type of a value. Because hypertables will be used, NOT NULL or UNIQUE are not allowed. Uniqueness must be
       enforced by the index.

    :param value: A value.
    :param key: A key.
    :param time_col: The name of the time column. Default is "time".
    :return: It returns a string with the SQL type. Like "BIGINT" or "TEXT".
    """
    if key == time_col:
        # column_type = "TIMESTAMPTZ NOT NULL"  # comentado pq se ajusta la unicidad al pasar a hipertabla
        column_type = "TIMESTAMPTZ"
    elif type(value) is bool:
        column_type = "BOOLEAN"
    elif type(value) is int:
        column_type = "BIGINT"
    elif type(value) is float:
        column_type = "DOUBLE PRECISION"
    else:
        column_type = "TEXT"
    sql_logger.debug(f"Data type for the column key={key} value={value}: {column_type}")
    return column_type


def check_unique_index_with_time(cursor, table_name: str, time_col: str) -> bool:
    """
    Checks if a unique index with the time column exists for the given table.

    :param cursor: A psycopg2 cursor connected to the database.
    :param table_name: A table name.
    :param time_col: A time column name.
    :return: Returns True if the unique index exists, False otherwise.
    """
    # Query para verificar si existe un índice único con 'time' para la tabla dada
    check_unique_index_query = f"""
    SELECT indexname, indexdef
    FROM pg_indexes
    WHERE tablename = '{table_name}'
    """

    cursor.execute(check_unique_index_query)
    indexes = cursor.fetchall()

    for index in indexes:
        index_name, index_definition = index
        if "UNIQUE" in index_definition:
            if time_col in index_definition:
                sql_logger.debug(f"Unique index {index_name} with '{time_col}' column found.")
                return True

    sql_logger.error(f"No unique index with '{time_col}' column found.")
    return False


def create_table_and_hypertable(cursor,
                                table_name: str,
                                column_definitions: dict,
                                time_col: str,
                                additional_index_column=None):
    """
    Creates a regular table and converts it to a hypertable. If the table already exists, it does nothing.

    :param cursor: A psycopg2 cursor connected to the database.
    :param table_name: Name of the table.
    :param column_definitions: Definitions of the columns. A dictionary where the key is the name of the column and the value is
    :param time_col: The name of the time column.
    :param additional_index_column: Any additional column to be indexed. If None, just time_col will be indexed.
    :return: It returns nothing.
    """

    # Guardar un SAVEPOINT
    cursor.execute("SAVEPOINT before_create_table;")

    try:
        # Crear la tabla regular primero
        columns_sql = ', '.join([f'"{k}" {infer_sql_type(v, k)}' for k, v in column_definitions.items()])
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_sql});"
        cursor.execute(create_table_sql)

        # Convertir la tabla en una hipertabla
        cursor.execute(f"SELECT create_hypertable('{table_name}', 'time');")
        if additional_index_column == time_col:
            additional_index_column = None

        if additional_index_column:
            cursor.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_{table_name} ON {table_name}(time, {additional_index_column});")
        else:
            cursor.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_{table_name} ON {table_name}(time);")

        # Si todo sale bien, liberar el SAVEPOINT
        cursor.execute("RELEASE SAVEPOINT before_create_table;")
        assert check_unique_index_with_time(cursor, table_name=table_name, time_col=time_col), ("No se pudo crear el índice único con "
                                                                                                "'time'.")

    except Exception as e:
        # Si algo sale mal, revertir al SAVEPOINT
        cursor.execute("ROLLBACK TO SAVEPOINT before_create_table;")
        raise e


def insert_data(cursor, table_name: str, records: List[dict], time_column: str, unique_column: str = None):
    """
    Inserts data into a table.

    :param cursor: A psycopg2 cursor connected to the database.
    :param table_name: Name of the table.
    :param records: A list of dictionaries with the data to insert.
    :param time_column: The name of the time column.
    :param unique_column: The name of the unique column. If None, no extra uniqueness will be enforced.
    :return: It returns nothing.
    """
    if time_column == unique_column:
        unique_column = None

    columns = records[0].keys()
    sql_logger.debug(f"Columnas de la tabla {table_name}: {columns}")
    data_to_insert = [tuple(record[col] for col in columns) for record in records]  # convierte a tuplas
    columns_without_unique = [col for col in columns if col != unique_column]

    if unique_column:
        insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT ({}, {}) DO UPDATE SET {}").format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.SQL(", ").join(sql.SQL("to_timestamp(%s / 1000.0)") if col == time_column else sql.Placeholder() for col in columns),
            sql.Identifier(time_column),
            sql.Identifier(unique_column),
            sql.SQL(", ").join(map(lambda col: sql.SQL(f'"{col}" = EXCLUDED."{col}"'), columns_without_unique)),
        )
    else:
        insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT ({}) DO UPDATE SET {}").format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.SQL(", ").join(sql.SQL("to_timestamp(%s / 1000.0)") if col == time_column else sql.Placeholder() for col in columns),
            sql.Identifier(time_column),
            sql.SQL(", ").join(map(lambda col: sql.SQL(f'"{col}" = EXCLUDED."{col}"'), columns_without_unique)),
        )

    sql_logger.debug(f"Query: {insert_query.as_string(cursor.connection)}")
    cursor.executemany(insert_query, data_to_insert)


def update_table_columns(cursor, table_name: str, record: dict, checked_columns: list) -> list:
    """
    Verificar igualdad de columnas e insertar columnas faltantes en su caso.

    :param cursor: Cursor de la conexión a la base de datos.
    :param table_name: Nombre de la tabla.
    :param record: Un diccionario con los datos a typo a insertar para deducir el tipo de datos.
    :param checked_columns: Lista de columnas ya verificadas.
    :return: Retorna una lista con las tablas verificadas actualizadas.
    """
    try:
        existing_columns = get_column_names(cursor=cursor, table_name=table_name, own_transaction=False)
        missing_columns = set(record.keys()) - set(existing_columns)
        if not missing_columns:
            checked_columns.append(table_name)
            return checked_columns
        else:
            for column in missing_columns:
                value = record[column]
                column_type = infer_sql_type(value=value, key=column)
                sql_logger.info(f"Insertando columna {column} de tipo {column_type} en la tabla {table_name}")
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} {column_type}")
                if not table_name in checked_columns:
                    checked_columns.append(table_name)
        return checked_columns

    except Exception as e:
        sql_logger.error(f"Error update_table_columns: {e}")
        raise e


def flexible_tables_and_data_insert(cursor,
                                    parsed_dict: Dict[str, List[dict]],
                                    verified_tables: list = None,
                                    checked_columns: list = None,
                                    time_column="time",
                                    batch=10) -> Tuple[list, list]:
    """
    Checks if the tables exist and creates them if they do not exist. Then insert the data into the tables.

    :param cursor: A psycopg2 cursor connected to the database.
    :param parsed_dict: A dictionary with the data to be inserted. The key is the name of the table and the value is a list of
      dictionaries with the data to be inserted.
    :param verified_tables: A set of verified tables.
    :param checked_columns: A set of checked columns.
    :param time_column: Name of the time column.
    :param batch: Size of the batch to insert data.
    :return: It returns a tuple with the verified tables and the checked columns.
    """
    if not verified_tables:
        verified_tables = []  # Caché de tablas verificadas
    if not checked_columns:
        checked_columns = []  # Caché de columnas verificadas
    try:
        # Iniciar transacción
        cursor.execute("BEGIN")
        cnt = 0
        for table_name, records in parsed_dict.items():

            if not records:  # records es una lista de diccionarios
                continue  # Saltar si la lista está vacía

            table_name = sanitize_table_name(table_name)
            data_type = data_type_from_table(table=table_name)

            # fuerza unicidad en una columna
            unique_column = stream_uniqueness_id_in_timescale[data_type]
            if unique_column == time_column:
                unique_column = None
            if table_name not in verified_tables:
                table_exists = is_hypertable(cursor, table_name=table_name)
                if table_exists:
                    verified_tables.append(table_name)
                else:
                    create_table_and_hypertable(cursor=cursor,
                                                table_name=table_name,
                                                column_definitions=records[0],
                                                time_col=time_column,
                                                additional_index_column=unique_column)

            checked_columns = update_table_columns(cursor=cursor, table_name=table_name, record=records[0], checked_columns=checked_columns)

            insert_data(table_name=table_name,
                        records=records,
                        cursor=cursor,
                        time_column=time_column,
                        unique_column=unique_column)

            if cnt % batch == 0:
                cursor.execute("COMMIT")
                sleep(0.1)
                cursor.execute("BEGIN")

        cursor.execute("COMMIT")

    except Exception as e:
        cursor.execute("ROLLBACK")
        sql_logger.error(f"Error durante la inserción de datos: {e}")
        raise e
    return verified_tables, checked_columns


def data_type_table(table_name: str) -> str:
    """
    Gets the data type of the table name.

    :param table_name: The name of the table.
    :return: Websockets channel type.
    """
    return data_type_from_table(table=table_name)


def delete_dupes(cursor,
                 table: str,
                 dupes: List[int],
                 column: str,
                 is_timestamp: bool,
                 ignore_errors: bool = False):
    """
    Delete a dupe by continuity column.

    :param cursor: A psycopg2 cursor connected to the database.
    :param table: The name of the table.
    :param dupes: A list with the values of the dupe.
    :param column: The name of the continuity column.
    :param is_timestamp: If True, convert the values to timestamp.
    :param ignore_errors: If True, ignore errors.
    :return: None
    """
    try:
        cursor.execute("BEGIN")
        pbar = tqdm(dupes)
        for dupe in pbar:
            pbar.set_description(f"Processing {table} {dupe}")
            delete_record(cursor=cursor,
                          table_name=table,
                          field_name=column,
                          value=dupe,
                          is_timestamp=is_timestamp,
                          own_transaction=False)
        cursor.execute("COMMIT")
    except Exception as e:
        # Si algo sale mal, revertir la transacción
        cursor.connection.rollback()
        msg = f"Error al eliminar duplicados en la tabla {table}: {e}"
        sql_logger.error(msg)
        if not ignore_errors:
            raise BinPanException(msg)


def data_type_from_table(table: str) -> str or None:
    """
    Gets the data type of the table name.

    :param table: The name of the table.
    :return: Websockets channel type.
    """
    if "kline" in table and not "missed" in table:
        return "kline"
    elif "aggTrade" in table or "aggtrade" in table:
        return "aggTrade"
    elif "trade" in table:
        return "trade"
    elif table.endswith("_statistics"):
        return "statistics"
    elif "missed" in table:
        return None
    else:
        sql_logger.debug(f"data_type_from_table: Name {table} not recognized.")
        return None


######################
# funciones atómicas #
######################


def check_standard_table_exists(cursor, table_name: str) -> bool:
    """
    Checks if a standard table exists.

    :param cursor: A psycopg2 cursor connected to the database.
    :param table_name: Name of the table.
    :return: Returns True if the table exists, False otherwise.
    """
    query = f"""
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_name = '{table_name}'
    );
    """
    cursor.execute(query)
    return cursor.fetchone()[0]


def get_indexed_columns(cursor, table_name) -> list:
    """
    No tengo muy claro la respuesta tan extendida de campos que da esta consulta.

    :param cursor: Un cursor de psycopg2 conectado a la base de datos.
    :param table_name: Nombre de la tabla.
    :return: Una lista con los nombres de las columnas indexadas.
    """
    query = f"""
    SELECT a.attname AS column_name
    FROM pg_class t, pg_class i, pg_index ix, pg_attribute a
    WHERE t.oid = ix.indrelid
        AND i.oid = ix.indexrelid
        AND a.attnum = ANY(ix.indkey)
        AND t.relkind = 'r'
        AND t.relname = '{table_name}';
    """
    cursor.execute(query)
    indexed_columns = [row[0] for row in cursor.fetchall()]
    return indexed_columns


def get_hypertable_indexes(cursor, hypertable_name, schema_name='public'):
    """
    Return example:

      .. code block:: python

        [('unfiusdt_trade_time_idx',
          'CREATE INDEX unfiusdt_trade_time_idx ON public.unfiusdt_trade USING btree ("time" DESC)'),
         ('idx_unfiusdt_trade_trade_id',
          'CREATE INDEX idx_unfiusdt_trade_trade_id ON public.unfiusdt_trade USING btree ("time", trade_id)')]

    :param cursor: A psycopg2 cursor connected to the database.
    :param hypertable_name: Name of the hypertable.
    :param schema_name: Name of the schema. Default is 'public'.
    :return: A list of tuples with the index name and the index definition.
    """
    query = f"""
    SELECT indexname, indexdef 
    FROM pg_indexes 
    WHERE tablename = '{hypertable_name}' 
    AND schemaname = '{schema_name}';
    """
    cursor.execute(query)
    hypertable_indexes = cursor.fetchall()
    return hypertable_indexes


def add_index_to_hypertable(cursor, hypertable_name: str, column_name: str, index_name: str = None):
    """
    Añade un índice a una hypertable en TimescaleDB.

    Ejemplo:

     .. code-block:: python

        add_index_to_hypertable(cursor, "simple_table", column_name="time", index_name="pepe")

        [('simple_table_time_idx',
          'CREATE INDEX simple_table_time_idx ON public.simple_table USING btree ("time" DESC)'),
         ('pepe', 'CREATE INDEX pepe ON public.simple_table USING btree ("time")')]

    :param cursor: Un cursor de psycopg2 conectado a la base de datos.
    :param hypertable_name: Nombre de la hypertable.
    :param column_name: Nombre de la columna a la que se añadirá el índice.
    :param index_name: Nombre opcional para el nuevo índice. Si no se proporciona, se generará automáticamente.
    :return: Retorna True si la operación fue exitosa, False en caso contrario.

    """
    try:
        if index_name is None:
            index_name = f"{hypertable_name}_{column_name}_idx"
        create_index_query = f"""CREATE INDEX {index_name} ON {hypertable_name} ({column_name});"""
        cursor.execute(create_index_query)
        return True
    except Exception as e:
        print(f"Error al añadir índice a la hypertable: {e}")
        return False


def check_index_exists(cursor, table_name, index_name):
    query = f"""
    SELECT 1
    FROM timescaledb_information.hypertable_indexes
    WHERE hypertable_name = '{table_name}'
    AND index_name = '{index_name}';
    """
    cursor.execute(query)
    return cursor.fetchone() is not None


#####################
# convertir y crear #
#####################


def create_simple_table(cursor, table_name: str, columns: list):
    """
    Create a simple table in PostgreSQL.

    :param cursor: A psycopg2 cursor connected to the database.
    :param table_name: Name of the new table.
    :param columns: List of tuples describing the columns. Each tuple must have the column name and the data type (e.g.
     [("id", "SERIAL PRIMARY KEY"), ("name", "VARCHAR(50)")]).

    """
    # Crear la definición de las columnas para la query SQL
    columns_definition = ", ".join([f'"{name}" {data_type}' for name, data_type in columns])

    # Query para crear la tabla
    create_table_query = f'CREATE TABLE "{table_name}" ({columns_definition});'

    # Ejecutar la query
    cursor.execute(create_table_query)


############################
# tablas de errores de api #
############################


def create_missed_table(cursor, continuity_field: str, miss_table: str):
    """
    Create a table to store the missed ids.

    :param continuity_field: A continuity field. Like "time" or "trade_id".
    :param cursor: A psycopg2 cursor connected to the database.
    :param miss_table: A table name. Like "ltcusdt_trade_missed" or "ltcusdt_kline_1m_missed".
    :return: None
    """
    if continuity_field == "time":
        column_def = {"time": 0}
        additional_index_column = None
    else:
        column_def = {"time": 0, continuity_field: 0}
        additional_index_column = continuity_field
    sql_logger.info(f"create_missed_table: Creating missed from API ids table {miss_table} with index 'time' and optionally:'"
                    f"{additional_index_column}'")
    create_table_and_hypertable(cursor=cursor,
                                table_name=miss_table,
                                column_definitions=column_def,
                                time_col="time",
                                additional_index_column=additional_index_column)


def get_missed_ids(cursor,
                   table: str,
                   continuity_field: str,
                   own_transaction: bool = True) -> List[int]:
    """
    Get the missed timestamps or trade_id from the database for each table. Each table has a parallel table with api misses.

    :param cursor: A psycopg2 cursor connected to the database.
    :param table: Missed ids table name. Default is "table_name_missed".
    :param continuity_field: The name of the continuity field.
    :param own_transaction: If True, the function will create its own transaction. If False, the function will use the
    :return: A list with the missed timestamps.
    """
    miss_table = f"{table}_missed"
    if own_transaction:
        cursor.execute("BEGIN")
    if not miss_table in list_tables_with_suffix(cursor=cursor, suffix="_missed"):
        create_missed_table(cursor=cursor, continuity_field=continuity_field, miss_table=miss_table)

    query = f'SELECT "{continuity_field}" FROM {miss_table} ORDER BY time;'

    cursor.execute(query)

    if continuity_field == "time":
        missed_ids = [int(row[0].timestamp() * 1000) for row in cursor.fetchall()]
    else:
        missed_ids = [int(row[0]) for row in cursor.fetchall()]

    if own_transaction:
        cursor.execute("COMMIT")

    return sorted(list(missed_ids))


def insert_missed_from_api_ids(cursor,
                               table: str,
                               misses: List[int],
                               continuity_field: str,
                               previously_used_missed_tables: list = None,
                               own_transaction: bool = True) -> list:
    """
    Insert missed ids from the API into the database.

    :param cursor: A psycopg2 cursor connected to the database.
    :param table: A table name. Like "ltcusdt_trade".
    :param misses: A list of missed ids. Each id is a timestamp or a trade_id. Example: [1620000000000, 1620000000001, ...]
    :param continuity_field: A continuity field. Like "time" or "trade_id".
    :param previously_used_missed_tables: A list of existing tables. If None, it will be retrieved from the database.
    :param own_transaction: A boolean. If True, the function will create its own transaction. If False, the function will use the
    :return: It returns nothing.
    """
    miss_table = f"{table}_missed"
    if own_transaction:
        cursor.execute("BEGIN")
    if not miss_table in previously_used_missed_tables:
        if not miss_table in list_tables_with_suffix(cursor=cursor, suffix="_missed"):
            create_missed_table(cursor=cursor, continuity_field=continuity_field, miss_table=miss_table)
        previously_used_missed_tables.append(miss_table)

    insertion = [{continuity_field: miss} for miss in misses]

    insert_data(cursor=cursor,
                table_name=miss_table,
                records=insertion,
                time_column="time",
                unique_column=continuity_field)
    if own_transaction:
        cursor.execute("COMMIT")
    return previously_used_missed_tables
