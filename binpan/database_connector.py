import pandas as pd

from handlers.postgresql import (create_connection, get_valid_table_list, is_cursor_alive, check_standard_table_exists,
                                 is_hypertable, get_column_names, get_indexed_columns, get_hypertable_indexes,
                                 list_tables_with_suffix, data_type_from_table, count_rows_in_tables)
from handlers import postgresql_database
from handlers.starters import AesCipher
from typing import List

cipher_object = AesCipher()


class Database:
    def __init__(self,
                 host: str = None,
                 port: str = 5432,
                 user: str = None,
                 password: str = None,
                 database: str = "crypto"):

        if not host:
            from secret import postgresql_host
            self.host = postgresql_host
        else:
            self.host = host

        if not port:
            from secret import postgresql_port
            self.port = int(postgresql_port)
        else:
            self.port = port

        if not user:
            from secret import postgresql_user
            self.user = postgresql_user
        else:
            self.user = user

        if not password:
            from secret import postgresql_password
            self.password = postgresql_password
        else:
            self.password = str(cipher_object.encrypt(password))

        if not database:
            from secret import postgresql_database
            self.database = postgresql_database
        else:
            self.database = database

        print(f"Host: {self.host}\nPort: {self.port}\nUser: {self.user}\nDatabase: {self.database}")

        # chequear types de las variables
        self.connection, self.cursor = create_connection(user=self.user,
                                                         host=self.host,
                                                         enc_password=self.password,
                                                         port=self.port,
                                                         database=self.database)

        self.tables = self.get_tables()
        self.size = self.get_db_size()
        self.table_sizes = self.get_tables_sizes(only_public_schema=True)
        self.hypertable_info = self.get_hypertable_info()

    def update_cursor(self):
        self.connection, self.cursor = create_connection(user=self.user,
                                                         enc_password=self.password,
                                                         host=self.host,
                                                         port=int(self.port),
                                                         database=self.database)
        self.status()

    def get_columns(self, table_name: str):
        return get_column_names(cursor=self.cursor, table_name=table_name, own_transaction=True)

    def close(self):
        self.connection.close()

    def status(self):
        print(f"Host: {self.host}\nPort: {self.port}\nUser: {self.user}\nPassword: XXXXXXXXXXXX \nDatabase: "
              f"{self.database}\nConnection: {self.connection}\nCursor: {self.cursor}")

    def get_tables(self, table_type: str = None, raw=False):
        if raw:
            return list_tables_with_suffix(self.cursor, "")
        elif table_type:
            return list_tables_with_suffix(self.cursor, table_type)
        else:
            return get_valid_table_list(self.cursor)

    def get_tables_sizes(self, only_public_schema: bool = True):
        self.table_sizes = postgresql_database.get_table_sizes(cursor=self.cursor, only_public_schema=only_public_schema)
        return self.table_sizes

    def get_db_size(self):
        self.size = postgresql_database.get_db_size(cursor=self.cursor)
        return self.size

    def get_hypertable_info(self):
        self.hypertable_info = postgresql_database.get_hypertable_info(cursor=self.cursor)
        return self.hypertable_info

    def alive(self):
        return is_cursor_alive(self.cursor)

    def table_config(self, table_name: str):
        exists = check_standard_table_exists(cursor=self.cursor, table_name=table_name)
        hyper = is_hypertable(cursor=self.cursor, table_name=table_name)
        columns = get_column_names(cursor=self.cursor, table_name=table_name, own_transaction=True)
        index = get_indexed_columns(cursor=self.cursor, table_name=table_name)
        hyper_index = get_hypertable_indexes(cursor=self.cursor, hypertable_name=table_name)

        # mostrar una tabla con todos los datos recopilados
        print(f"Table: {table_name}\nExists: {exists}\nHyper: {hyper}\nColumns: {columns}\nIndex: {index}\nHyper Index: {hyper_index}")

    def get_table_counts(self, tables: List[str] = None) -> pd.Series:
        if not tables:
            tables = self.tables
        ret = count_rows_in_tables(cursor=self.cursor, table_names=tables)
        return pd.Series(ret).sort_values(ascending=False)

    @staticmethod
    def table_type(table_name: str):
        data_type = data_type_from_table(table_name)
        return data_type
