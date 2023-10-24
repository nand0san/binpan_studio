import pandas as pd


def get_db_size(cursor) -> str:
    """
    Get the size of the database in a human readable format

    :param cursor: A psycopg2 cursor
    :return: A string with the size of the database
    """
    cursor.execute("SELECT pg_size_pretty(pg_database_size(current_database())) AS db_size;")
    return cursor.fetchone()[0]


def get_table_sizes(cursor, only_public_schema=False) -> pd.DataFrame:
    """
    Get the size of each table in the database and return it as a pandas DataFrame with sizes in KB.

    :param cursor: A psycopg2 cursor
    :param only_public_schema: Optional, set to True to only get sizes of tables in the 'public' schema.
    :return: A pandas DataFrame with the size of each table in the database
    """
    base_query = """
    SELECT 
        schemaname,
        tablename,
        (pg_table_size(schemaname || '.' || tablename)/1024) AS table_size_kb,
        (pg_indexes_size(schemaname || '.' || tablename)/1024) AS indexes_size_kb,
        (pg_total_relation_size(schemaname || '.' || tablename)/1024) AS total_size_kb
    FROM pg_tables
    """

    # Add condition for filtering by 'public' schema if needed
    if only_public_schema:
        base_query += " WHERE schemaname = 'public' "

    base_query += " ORDER BY total_size_kb DESC; "

    cursor.execute(base_query)
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame([dict(zip(columns, row)) for row in cursor.fetchall()])

    # Convertir las columnas de tamaño a numéricas
    for col in ["table_size_kb", "indexes_size_kb", "total_size_kb"]:
        df[col] = df[col].astype(float)

    return df.sort_values("total_size_kb", ascending=False)


def get_wal_gb_size(cursor) -> float:
    """
    Get the size of the Write Ahead Log (WAL) in the database and return it in GB.

    :param cursor: a psycopg2 cursor
    :return: Size in GB of the WAL
    """
    query = """
    SELECT 
        (pg_current_wal_lsn() - '0/00000000'::pg_lsn) / 1024 / 1024 / 1024 AS wal_size_gb;
    """
    cursor.execute(query)
    wal_gb = cursor.fetchone()[0]
    return float(wal_gb)


def get_active_connections(cursor, database_name="crypto", include_superuser=True) -> int:
    """
    Get the number of active connections in the database.

    :param cursor: a psycopg2 cursor
    :param database_name: Optional, name of the specific database to filter.
    :param include_superuser: Optional, whether to include superuser connections or not.
    :return: Number of active connections
    """
    base_query = """
    SELECT COUNT(*) 
    FROM pg_stat_activity 
    WHERE state = 'active' 
    """

    filters = []
    if database_name:
        filters.append(f"datname = '{database_name}'")
    if not include_superuser:
        filters.append("usesysid != 10")

    if filters:
        base_query += "AND " + " AND ".join(filters)

    cursor.execute(base_query)
    active_connections = cursor.fetchone()[0]
    return active_connections


def get_autovacuum_level(cursor) -> pd.DataFrame:
    """
    Get the Autovacuum status for tables in the database.

    This function returns a DataFrame with the following columns:

    - `schemaname`:
       The schema where the table resides.

    - `relname`:
       The name of the table.

    - `last_autovacuum`:
       Timestamp of the last autovacuum operation performed on the table.

    - `last_autoanalyze`:
       Timestamp of the last autoanalyze operation performed on the table.

    - `n_dead_tup`:
       Number of dead tuples in the table. Dead tuples are rows that have been updated or deleted and are awaiting removal by autovacuum.

    - `n_live_tup`:
       Number of live tuples in the table. Live tuples are rows that are currently valid and not marked for deletion.

    - `n_mod_since_analyze`:
       Number of tuples modified since the last analyze operation. Analyze operations collect statistics about the data in the table
       to help the query planner optimize queries.

    - `seq_scan`:
       Number of sequential scans performed on the table. A high number can indicate that indexes are not being effectively utilized.

    - `idx_scan`:
       Number of index scans performed on the table. Indicates how often indexes are used for querying this table.

    :param cursor: a psycopg2 cursor
    :return: DataFrame with Autovacuum status
    """
    query = """
    SELECT 
        schemaname,
        relname,
        last_autovacuum,
        last_autoanalyze,
        n_dead_tup,
        n_live_tup,
        n_mod_since_analyze,
        seq_scan,
        idx_scan
    FROM pg_stat_user_tables;
    """
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame([dict(zip(columns, row)) for row in cursor.fetchall()])

    # Convertir columnas relevantes a numéricas
    numeric_cols = ["n_dead_tup", "n_live_tup", "n_mod_since_analyze", "seq_scan", "idx_scan"]
    for col in numeric_cols:
        df[col] = df[col].astype(float)

    return df


def get_ungranted_locks(cursor) -> pd.DataFrame:
    """
    Get the ungranted locks in the database.

    :param cursor: a psycopg2 cursor
    :return: DataFrame with ungranted lock details
    """
    query = """
    SELECT 
        pid,
        relation::regclass as relation_name,
        mode,
        granted
    FROM pg_locks
    WHERE NOT granted;
    """
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame([dict(zip(columns, row)) for row in cursor.fetchall()])

    return df


def get_cache_statistics(cursor) -> pd.DataFrame:
    """
    Get cache statistics for tables in the database.

    This function returns a DataFrame with the following columns:

    - `table_name`:
       The name of the table.

    - `heap_blks_read`:
       Number of disk blocks read for the main table (heap).

    - `heap_blks_hit`:
       Number of buffer hits in the cache for the main table (heap).

    - `idx_blks_read`:
       Number of disk blocks read for all indexes on the table.

    - `idx_blks_hit`:
       Number of buffer hits in the cache for all indexes on the table.

    - `toast_blks_read`:
       Number of disk blocks read for the TOAST table (used for storing large values out of main table rows).

    - `toast_blks_hit`:
       Number of buffer hits in the cache for the TOAST table.

    - `tidx_blks_read`:
       Number of disk blocks read for indexes on the TOAST table.

    - `tidx_blks_hit`:
       Number of buffer hits in the cache for indexes on the TOAST table.

    - `heap_cache_hit_rate`:
       Cache hit rate for the main table (heap).

    - `idx_cache_hit_rate`:
       Cache hit rate for all indexes on the table.

    Cache hit rates are useful metrics for determining the efficiency of the cache. A high hit rate
    indicates that the cache is effectively reducing the need for disk reads.

    :param cursor: a psycopg2 cursor
    :return: DataFrame with cache statistics
    """
    query = """
    SELECT 
        relname AS table_name,
        heap_blks_read,
        heap_blks_hit,
        idx_blks_read,
        idx_blks_hit,
        toast_blks_read,
        toast_blks_hit,
        tidx_blks_read,
        tidx_blks_hit
    FROM pg_statio_user_tables;
    """
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame([dict(zip(columns, row)) for row in cursor.fetchall()])

    # Convertir las columnas a numéricas
    numeric_cols = [
        "heap_blks_read", "heap_blks_hit",
        "idx_blks_read", "idx_blks_hit",
        "toast_blks_read", "toast_blks_hit",
        "tidx_blks_read", "tidx_blks_hit"
    ]
    for col in numeric_cols:
        df[col] = df[col].astype(float)

    # Si deseas agregar una columna con el porcentaje de éxito del caché:
    df["heap_cache_hit_rate"] = df["heap_blks_hit"] / (df["heap_blks_hit"] + df["heap_blks_read"])
    df["idx_cache_hit_rate"] = df["idx_blks_hit"] / (df["idx_blks_hit"] + df["idx_blks_read"])

    return df


def get_hypertable_info(cursor) -> pd.DataFrame:
    """
    Get information about hypertables in the TimescaleDB.

    This function returns a DataFrame with the following columns:

    - `hypertable_schema`:
       The schema in which the hypertable resides.

    - `hypertable_name`:
       The name of the hypertable.

    - `owner`:
       The owner of the hypertable.

    - `num_dimensions`:
       The number of dimensions of the hypertable.

    - `num_chunks`:
       The total number of chunks associated with the hypertable.

    - `compression_enabled`:
       Whether compression is enabled for the hypertable.

    - `is_distributed`:
       Indicates if the hypertable is distributed.

    - `replication_factor`:
       The replication factor of the hypertable (relevant for distributed hypertables).

    - `data_nodes`:
       Nodes where the hypertable data resides.

    - `tablespaces`:
       Tablespaces associated with the hypertable.

    :param cursor: a psycopg2 cursor
    :return: DataFrame with hypertable details
    """
    query = """
    SELECT 
        hypertable_schema,
        hypertable_name,
        owner,
        num_dimensions,
        num_chunks,
        compression_enabled,
        is_distributed,
        replication_factor,
        data_nodes,
        tablespaces
    FROM timescaledb_information.hypertables;
    """
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame([dict(zip(columns, row)) for row in cursor.fetchall()])

    return df.sort_values("num_chunks", ascending=False)
