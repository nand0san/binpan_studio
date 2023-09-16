import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client import Point

from datetime import datetime, timedelta
from .logs import Logs
from typing import List, Dict

influx_logger = Logs(filename='./logs/influx_manager.log', name='influx_manager', info_level='DEBUG')


def time_clause(timestamp_ms: int) -> str or None:
    """
    If a timestamp in string (delta) or int, then convert it to a delta string or ISO format.

    :param timestamp_ms: A timestamp in milliseconds. If str passed, expected format is ISO or Influx Delta: 1h, -1d, etc...
    :return: A string in ISO format or delta.
    """
    if timestamp_ms is None:
        return None
    elif isinstance(timestamp_ms, str):
        influx_logger.debug(f"Timestamp is already a string: {timestamp_ms}")
        return timestamp_ms
    try:
        # test of unix milliseconds timestamp
        assert isinstance(timestamp_ms, int), f"Timestamp must be an integer, not {type(timestamp_ms)}"
        assert timestamp_ms > 0, f"Timestamp must be positive, not {timestamp_ms}"
        assert len(str(timestamp_ms)) == 13, f"Timestamp must be in milliseconds, not {timestamp_ms}"
        return datetime.utcfromtimestamp(timestamp_ms / 1000.0).isoformat() + "Z"
    except AssertionError as exc:
        influx_logger.error(f"Error converting milliseconds to ISO: {timestamp_ms} ---> {exc}")
        return None


##########
# CLIENT #
##########


def influx_client(url: str,
                  token: str,
                  org: str,
                  timeout: int = 10000,
                  ) -> influxdb_client.InfluxDBClient:
    """
    Función para obtener un cliente de InfluxDB en modo lectura.

    :param str url: Url de InfluxDB. Por ejemplo, "http://localhost:8086"
    :param str token: Token de acceso a InfluxDB.
    :param str org: Organización de InfluxDB. Por ejemplo, "tu_org"
    :param int timeout: Timeout de la conexión en milisegundos.
    :return: Un cliente de InfluxDB.
    """
    return influxdb_client.InfluxDBClient(url=url,
                                          token=token,
                                          org=org,
                                          debug=False,
                                          timeout=timeout)


#########
# FETCH #
#########


def fetch_measurements(client: influxdb_client.InfluxDBClient,
                       bucket: str,
                       org: str) -> set:
    """
    Función para obtener todas las medidas de un bucket específico.

    :param client: Cliente conectado a InfluxDB.
    :param bucket: Bucket de InfluxDB.
    :param org: Organización de InfluxDB.
    :return: Lista de medidas.
    """
    query_api = client.query_api()
    query = f'import "influxdata/influxdb/schema"\n' \
            f'schema.measurements(bucket: "{bucket}")'
    result = query_api.query(org=org, query=query)
    return {i.get_value() for i in result[0].records}


def fetch_measurement_tag_keys(client: influxdb_client.InfluxDBClient,
                               bucket: str,
                               measurement: str,
                               org: str) -> set:
    """
    Función para obtener todos los tags de una medida específica en un bucket específico.

    :param client: Cliente conectado a InfluxDB
    :param bucket: Nombre del bucket
    :param measurement: Nombre de la medida. Por ejemplo, "trades"
    :param org: Nombre de la organización. Por ejemplo, "tu_org"
    :return: Conjunto de tags únicos. Ejemplo de retorno:

    .. code-block:: python

        {'buyer_is_maker', 'stream', 'symbol'}

    """
    query_api = client.query_api()
    query = f'import "influxdata/influxdb/schema"\n' \
            f'schema.tagKeys(bucket: "{bucket}", predicate: (r) => r._measurement == "{measurement}")'
    result = query_api.query(org=org, query=query)
    tags = set()
    for table in result:
        for record in table.records:
            tags.add(record.get_value())
    return {i for i in tags if not i.startswith("_")}


def fetch_measurement_field_keys(client: influxdb_client.InfluxDBClient,
                                 bucket: str,
                                 measurement: str,
                                 org: str) -> set:
    """
    Función para obtener todos los campos de una medida específica en un bucket específico.

    :param client: Cliente conectado a InfluxDB
    :param bucket: Nombre del bucket
    :param measurement: Nombre de la medida. Por ejemplo, "trades"
    :param org: Nombre de la organización. Por ejemplo, "tu_org"
    :return: Conjunto de campos únicos. Ejemplo de retorno:

    .. code-block:: python

        {'buyer_order_id',
         'event_time',
         'price',
         'quantity',
         'quote_quantity',
         'seller_order_id',
         'trade_id',
         'trade_time'}

    """
    query_api = client.query_api()

    query = f'import "influxdata/influxdb/schema"\n' \
            f'schema.fieldKeys(bucket: "{bucket}", predicate: (r) => r._measurement == "{measurement}")'

    result = query_api.query(org=org, query=query)

    fields = set()
    for table in result:
        for record in table.records:
            fields.add(record.get_value())

    return fields


def fetch_measurement_tag_values(client: influxdb_client.InfluxDBClient,
                                 measurement: str,
                                 tag: str,
                                 bucket: str,
                                 org: str,
                                 start_time: int = None,
                                 end_time: int = None
                                 ) -> set:
    """
    Función para obtener todos los valores que tiene un tag en una medida específica en un bucket específico en el periodo de
    tiempo especificado.

    :param client: Cliente conectado a InfluxDB
    :param measurement: Nombre de la medida. Por ejemplo, "trades"
    :param tag: Nombre del tag. Por ejemplo, "stream"
    :param bucket: Nombre del bucket
    :param org: Nombre de la organización. Por ejemplo, "tu_org"
    :param start_time: Timestamp en milisegundos para el inicio del rango de datos. Examples: 1631000000000, "2021-09-07T00:00:00Z", "-1d"
    :param end_time: Timestamp en milisegundos para el final del rango de datos. Examples: 1631000000000, "2021-09-07T00:00:00Z", "-1d"
    :return: Conjunto de símbolos únicos. Ejemplo de retorno:

    .. code-block:: python

        streams = fetch_tag_values(client=client, bucket=influx_bucket, measurement="trades", org=influx_org, tag="stream")
        print(streams)

        {'duskusdt@trade', 'dotusdt@trade', 'multiusdt@trade', 'fluxusdt@trade', 'scrtusdt@trade', 'frontusdt@trade', 'tfuelusdt@trade',
         'hftusdt@trade', 'alcxusdt@trade', 'rsrusdt@trade', 'joeusdt@trade', 'tkousdt@trade', 'vthousdt@trade', 'galusdt@trade',
          'osmousdt@trade','...}
    """

    if start_time:
        start_time = time_clause(start_time)
    if end_time:
        end_time = time_clause(end_time)

    range_clause = f'  |> range(start: {start_time or "-30d"}, stop: {end_time or "now()"})'

    query_api = client.query_api()
    query = f'from(bucket: "{bucket}") {range_clause}' \
            f'  |> filter(fn: (r) => r["_measurement"] == "{measurement}")' \
            f'  |> distinct(column: "{tag}")'

    influx_logger.debug(f"Fetch Measurement Tag Values Query: {query}")
    result = query_api.query(org=org, query=query)

    # unique
    unique_streams = set()
    for table in result:
        for record in table.records:
            unique_streams.add(record.get_value())
    return unique_streams


def fetch_data(client: influxdb_client.InfluxDBClient,
               bucket: str,
               measurement: str,
               org: str,
               tags_filter: dict = {},
               fields_in_ret: list = [],
               tags_in_ret: list = [],
               start_time=None,
               end_time=None) -> list:
    """
    Función para obtener datos de una medida específica en un bucket específico, filtrado por tags y fields específicos.

    :param client: Cliente conectado a InfluxDB
    :param bucket: Nombre del bucket
    :param measurement: Nombre de la medida. Por ejemplo, "trades"
    :param org: Nombre de la organización. Por ejemplo, "tu_org"
    :param tags_filter: Diccionario de tags para filtrar. Por ejemplo, {"symbol": "BTC", "stream": "trades"}
    :param fields_in_ret: Lista de fields para seleccionar. Por ejemplo, ["field1", "field2"]
    :param tags_in_ret: Lista de tags para incluir en los resultados. Por ejemplo, ["stream"]
    :param start_time: Timestamp en milisegundos para el inicio del rango de datos. Examples: 1631000000000, "2021-09-07T00:00:00Z", "-1d"
    :param end_time: Timestamp en milisegundos para el final del rango de datos. Examples: 1631000000000, "2021-09-07T00:00:00Z", "-1d"
    :return: Lista de resultados. Ejemplo de retorno:

    .. code-block:: python

        data = fetch_data(client=client,
                          bucket=influx_bucket,
                          measurement="trades",
                          org=influx_org,
                          tags={"stream": "btcusdt@trade", "buyer_is_maker": True},
                          fields=fields,
                          tag_keys=tags)
        [{'result': '_result',
          'table': 0,
          '_time': datetime.datetime(2023, 9, 14, 16, 0, 2, 454000, tzinfo=tzutc()),
          '_measurement': 'trades',
          'buyer_is_maker': 'True',
          'stream': 'btcusdt@trade',
          'buyer_order_id': 22340720310,
          'price': 26693.97,
          'quantity': 0.00085,
          'quote_quantity': 22.6898745,
          'seller_order_id': 22340720976,
          'trade_id': 3212341083},
          ...]

    """
    if not fields_in_ret:
        fields_in_ret = fetch_measurement_field_keys(client=client, bucket=bucket, measurement=measurement, org=org)
    if not tags_in_ret:
        tags_in_ret = fetch_measurement_tag_keys(client=client, bucket=bucket, measurement=measurement, org=org)

    if start_time:
        start_time = time_clause(start_time)
    if end_time:
        end_time = time_clause(end_time)

    range_clause = f'  |> range(start: {start_time or "-1h"}, stop: {end_time or "now()"})'

    query_api = client.query_api()

    tag_filters = " and ".join([f'r["{key}"] == "{value}"' for key, value in tags_filter.items()])
    field_selection = ', '.join([f'"{field}"' for field in fields_in_ret]) if fields_in_ret else None
    tag_selection = ', '.join([f'"{tag}"' for tag in tags_in_ret]) if tags_in_ret else None

    additional_fields = ''

    if field_selection:
        additional_fields += ', ' + field_selection

    if tag_selection:
        additional_fields += ', ' + tag_selection

    if tag_filters:
        query = f'from(bucket: "{bucket}") {range_clause}' \
                f'  |> filter(fn: (r) => r["_measurement"] == "{measurement}" and {tag_filters})' \
                f'  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")' \
                f'  |> keep(columns: ["_time", "_measurement"{additional_fields}])'
    else:
        query = f'from(bucket: "{bucket}") {range_clause}' \
                f'  |> filter(fn: (r) => r["_measurement"] == "{measurement}")' \
                f'  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")' \
                f'  |> keep(columns: ["_time", "_measurement"{additional_fields}])'

    influx_logger.debug(f"Fetch Data Query: {query}")

    result = query_api.query(org=org, query=query)
    data = []
    for table in result:
        for record in table.records:
            data.append(record.values)
    return data


##########
# DELETE #
##########

def delete_measurement(client: influxdb_client.InfluxDBClient,
                       org: str,
                       bucket: str,
                       measurement: str,
                       days: int = 30):
    """
    Función para eliminar un measurement en InfluxDB 2.x. No se borra directamente, sino que se borran los datos. El proceso es asíncrono,
    por lo que puede tardar un tiempo en completarse.

    :param client: Cliente conectado a InfluxDB.
    :param org: Organización de InfluxDB.
    :param bucket: Bucket de InfluxDB.
    :param measurement: Nombre de la medida a eliminar.
    :param days: Número de días a eliminar.
    :return: None
    """

    start_time = (datetime.utcnow() - timedelta(days=days)).isoformat("T") + "Z"
    stop_time = datetime.utcnow().isoformat("T") + "Z"
    delete_api = client.delete_api()
    query = f'_measurement="{measurement}"'

    influx_logger.debug(f"Deleting measurement: {measurement} with query: {query}")
    influx_logger.debug(f"Start time: {start_time} to stop time: {stop_time}")

    delete_api.delete(start=start_time, stop=stop_time, predicate=query, bucket=bucket, org=org)


def delete_entries_by_tag_value(client: influxdb_client.InfluxDBClient,
                                org: str,
                                bucket: str,
                                measurement: str,
                                tag_key: str,
                                tag_value: str,
                                days: int = 30):
    """
    Función para eliminar entradas en un measurement específico donde un tag tiene un valor específico.

    :param client: Cliente conectado a InfluxDB.
    :param org: Organización de InfluxDB.
    :param bucket: Bucket de InfluxDB.
    :param measurement: Nombre de la medida a eliminar.
    :param tag_key: Nombre del tag.
    :param tag_value: Valor del tag.
    :param days: Número de días a eliminar.
    :return: None
    """
    start_time = (datetime.utcnow() - timedelta(days=days)).isoformat("T") + "Z"
    stop_time = datetime.utcnow().isoformat("T") + "Z"

    delete_api = client.delete_api()
    query = f'_measurement="{measurement}" and {tag_key}="{tag_value}"'

    influx_logger.debug(f"Deleting entries with query: {query}")
    influx_logger.debug(f"Start time: {start_time} to stop time: {stop_time}")

    delete_api.delete(start=start_time, stop=stop_time, predicate=query, bucket=bucket, org=org)


##########
# INSERT #
##########


def create_bulk_points(client: influxdb_client.InfluxDBClient,
                       bucket: str,
                       org: str,
                       measurement: str,
                       data_list: List[Dict]):
    """
    Función para crear una nueva medida en InfluxDB 2.x.

    :param client: Cliente conectado a InfluxDB.
    :param bucket: Bucket de InfluxDB.
    :param org: Organización de InfluxDB.
    :param measurement: Medida a crear.
    :param data_list: Datos a insertar. En formato de lista de diccionarios. Por ejemplo:

        .. code-block:: python

            data_list = [
                {"fields": {"field1": 1, "field2": 2},
                 "tags": {"tag1": "tag1", "tag2": "tag2"},
                 "time": 1234567890000000000}
            ]

    :return: None
    """

    points = []
    for data in data_list:
        point = Point(measurement)
        if "fields" in data:
            for field, value in data["fields"].items():
                point.field(field, value)
        if "tags" in data:
            for tag_key, tag_value in data["tags"].items():
                point.tag(tag_key, tag_value)
        if "time" in data:  # time in nanoseconds
            point.time(data["time"])
        points.append(point)

    write_api = client.write_api(write_options=SYNCHRONOUS)
    write_api.write(bucket=bucket, org=org, record=points, )
    write_api.__del__()
