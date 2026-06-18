import pandas as pd

from .storage.postgresql import (create_connection, get_valid_table_list, is_cursor_alive, check_standard_table_exists,
                                 is_hypertable, get_column_names, get_indexed_columns, get_hypertable_indexes,
                                 list_tables_with_suffix, data_type_from_table, count_rows_in_tables)
from .storage import postgresql_database
from .core.secrets import get_secret


class Database:
    def __init__(self,
                 host: str = None,
                 port: str = 5432,
                 user: str = None,
                 password: str = None,
                 database: str = "crypto"):

        # Credenciales gestionadas por panzer (~/.panzer_creds) si no se pasan.
        self.host = host if host else get_secret("postgresql_host")
        self.port = int(port if port else get_secret("postgresql_port"))
        self.user = user if user else get_secret("postgresql_user")
        self.password = password if password else get_secret("postgresql_password")
        self.database = database if database else get_secret("postgresql_database")

        print(f"Host: {self.host}\nPort: {self.port}\nUser: {self.user}\nDatabase: {self.database}")

        # chequear types de las variables
        self.connection, self.cursor = create_connection(user=self.user,
                                                         host=self.host,
                                                         password=self.password,
                                                         port=self.port,
                                                         database=self.database)

        self.tables = self.get_tables()
        self.size = self.get_db_size()
        self.table_sizes = self.get_tables_sizes(only_public_schema=True)
        self.hypertable_info = self.get_hypertable_info()

    def update_cursor(self):
        self.connection, self.cursor = create_connection(user=self.user,
                                                         password=self.password,
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

    def get_table_counts(self, tables: list[str] = None, drop_missed: bool = True) -> pd.Series:
        if not tables:
            tables = self.tables
        ret = count_rows_in_tables(cursor=self.cursor, table_names=tables)
        ret = pd.Series(ret).sort_values(ascending=False)
        if drop_missed:
            ret = ret[~ret.index.str.contains("missed")]
        return ret

    @staticmethod
    def table_type(table_name: str):
        data_type = data_type_from_table(table_name)
        return data_type
