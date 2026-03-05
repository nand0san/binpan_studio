# CLAUDE.md - Guía del proyecto BinPan Studio

## Descripción del proyecto

BinPan es un wrapper de Python para la API de Binance orientado al análisis de datos de mercado (velas,
trades, indicadores técnicos, plotting). Publicado en PyPI como `binpan`. Licencia MIT.

**Autor**: nand0san

---

## Idioma y comunicación

- **Idioma principal**: Español. Commits, comentarios inline y comunicación en español.
- **Docstrings**: En inglés, formato Sphinx (`:param type name: descripción`). Mantener este criterio.
- **README y docs públicas**: En inglés (orientadas a PyPI y GitHub Pages).

---

## Estructura del repositorio

```
binpan_studio/
├── binpan/              # Paquete principal (Symbol, Wallet, Exchange, Database)
│   ├── symbol_manager.py   # Clase Symbol (clase principal, ~215K)
│   ├── exchange_manager.py # Clase Exchange
│   ├── wallet_manager.py   # Clase Wallet
│   ├── database_connector.py
│   └── auxiliar.py         # Utilidades auxiliares
├── handlers/            # Módulos de funciones (lógica de bajo nivel)
│   ├── exchange.py         # Llamadas a API de exchange → SIMPLIFICAR con panzer
│   ├── market.py           # Llamadas a API de mercado → SIMPLIFICAR con panzer
│   ├── indicators.py       # Indicadores técnicos propios (mantener)
│   ├── plotting.py         # Plots con Plotly (mantener)
│   ├── tags.py             # Etiquetado y backtesting (mantener)
│   ├── time_helper.py      # Utilidades de tiempo → REDUCIR con kline-timestamp
│   ├── files.py            # Lectura/escritura de archivos (mantener)
│   ├── quest.py            # Peticiones HTTP a API → ELIMINAR parte pública (panzer)
│   ├── standards.py        # Constantes y mapeos de columnas (mantener)
│   ├── logs.py             # LogManager (logging rotativo) (mantener)
│   ├── exceptions.py       # Excepciones personalizadas (mantener)
│   ├── aggregations.py     # Resampleo de klines (mantener)
│   ├── postgresql.py       # Conector PostgreSQL (mantener)
│   ├── numba_tools.py      # Indicadores acelerados con Numba (mantener)
│   └── ...
├── objects/             # Clases de dominio
│   ├── timestamps.py       # ELIMINAR → reemplazar por kline-timestamp
│   ├── timeframes.py       # SIMPLIFICAR → usar KlineTimestamp internamente
│   ├── trades.py           # Clase Trades (hereda de Timeframe)
│   └── api.py              # ApiClient básico → ELIMINAR (panzer)
├── docs/                # Documentación Sphinx
├── setup.py             # Empaquetado (PyPI)
├── requirements.txt     # Dependencias
└── secret.py            # Claves API encriptadas (NUNCA commitear)
```

---

## Git: ramas, remotos y flujo de trabajo

### Ramas

| Rama   | Propósito |
|--------|-----------|
| `dev`  | **Rama principal de desarrollo**. Todo el trabajo diario se hace aquí. |
| `main` | Rama estable / publicación. Solo se actualiza con squash merges desde `dev`. |
| `simp` | **DEPRECADA**. La simplificación se aplica directamente en `dev`. |

**Trabajar siempre en `dev`**. No hacer commits directos en `main`.

### Remotos

| Remoto   | URL | Uso |
|----------|-----|-----|
| `nas`    | `ssh://ffad@192.168.89.201:22/.../binpan_studio_dev` | Backup completo en NAS local. Se puede pushear libremente sin riesgos de privacidad. Push normal (no squash). |
| `origin` | *(GitHub, cuando se configure)* | Repositorio público. **Siempre pushear con squash** para proteger privacidad del historial de commits. |

### Reglas de push

- **A `nas`**: push normal (`git push nas dev`). Sin restricciones de privacidad, es backup local.
- **A GitHub (`origin`)**: **siempre squash merge** o `git merge --squash` antes de pushear a `main`.
  Esto comprime el historial de desarrollo y evita exponer commits intermedios con información sensible.
- **Nunca pushear `secret.py`** ni archivos con claves API a ningún remoto. Ya está en `.gitignore`.

### Flujo típico

```bash
# Desarrollo diario
git checkout dev
# ... trabajar, commitear ...
git push nas dev          # backup al NAS (seguro)

# Publicar a GitHub
git checkout main
git merge --squash dev
git commit -m "descripción del release"
git push origin main      # push limpio a GitHub
git checkout dev           # volver a desarrollo
```

---

## Estilo de código Python

### Convenciones generales

- **Python >= 3.10** requerido.
- **snake_case** para funciones y variables.
- **CamelCase** para clases (`Symbol`, `Timeframe`, `Timestamp`, `Trades`, `LogManager`).
- **UPPER_CASE** para constantes (`REQUIRED`, `README`).
- **Type hints** con builtins de Python 3.10+ (ver sección Imports más abajo).
- Separadores visuales con comentarios `#####` para secciones dentro de módulos.

### Docstrings - Formato Sphinx

Todas las funciones y clases documentadas deben usar **Sphinx-style docstrings** en inglés:

```python
def mi_funcion(param1: str, param2: int = 10) -> pd.DataFrame:
    """
    Breve descripción de lo que hace.

    Descripción más extensa si es necesario.

    :param str param1: Descripción del primer parámetro.
    :param int param2: Descripción del segundo parámetro. Default 10.
    :return pd.DataFrame: Descripción del retorno.

    Example:

        .. code-block:: python

            resultado = mi_funcion("test", param2=5)
    """
```

- Usar `.. code-block:: python` para ejemplos de código.
- Usar `.. image::` para imágenes en la documentación.
- Las clases de `binpan/` se documentan con `.. autoclass::` en los `.rst`.
- Los módulos de `handlers/` se documentan con `.. automodule::` en los `.rst`.

### Imports - Política estricta

#### Principio general: código honesto, sin fallbacks

Si una dependencia no está instalada, **que dé error claro**. No enmascarar ImportErrors
con try/except ni asignar `None` como fallback. El usuario necesita saber qué falta.

```python
# MAL - fallback silencioso
try:
    from redis import StrictRedis
except ImportError:
    StrictRedis = None  # falla silenciosa en runtime

# BIEN - error claro al importar
from redis import StrictRedis  # si no está, ImportError inmediato y claro
```

#### Imports lazy (diferidos) para librerías pesadas

Para mejorar el tiempo de `import binpan` en notebooks y agentes, las librerías pesadas
que solo se usan en funciones específicas deben importarse **dentro de la función**, no
al nivel del módulo. Esto es especialmente importante para:

- `plotly` (solo en funciones de plotting)
- `scipy` / `scikit-learn` (solo en funciones estadísticas)
- `numba` (solo en funciones aceleradas)
- `psycopg2` (solo si se usa PostgreSQL)
- `influxdb_client` (solo si se usa InfluxDB)
- `redis` (solo si se usa Redis)

```python
# MAL - importar plotly al cargar el módulo
import plotly.graph_objects as go
import plotly.subplots as sp

def plot_candles(df):
    fig = sp.make_subplots(...)

# BIEN - import lazy dentro de la función
def plot_candles(df):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(...)
```

**Excepción**: `pandas` y `numpy` son el core del proyecto y se usan en casi todas las
funciones. Estos sí se importan a nivel de módulo.

#### Type hints: usar builtins de Python 3.10+

**NO usar** `typing.Tuple`, `typing.List`, `typing.Dict`, `typing.Set`. Están deprecados
desde Python 3.9 y el proyecto requiere Python >= 3.10. Usar los builtins directamente:

```python
# MAL (deprecado)
from typing import Tuple, List, Dict, Optional, Union
def mi_funcion(datos: List[str]) -> Tuple[int, str]: ...

# BIEN (Python 3.10+)
def mi_funcion(datos: list[str]) -> tuple[int, str]: ...
# Union se reemplaza con |
def otra_funcion(valor: str | int | None = None) -> pd.DataFrame | None: ...
```

Si el type hint solo se necesita para documentación estática, usar `from __future__ import annotations`
al inicio del archivo para que los hints sean strings y no se evalúen en runtime.

#### Prohibido `import *`

No usar `from modulo import *`. Importar siempre los nombres explícitamente.

```python
# MAL
from .auxiliar import *
from .standards import *

# BIEN
from .auxiliar import csv_klines_setup, check_continuity, repair_kline_discontinuity
from .standards import binance_api_candles_cols, agg_trades_api_map_columns
```

#### `handlers/__init__.py` - imports lazy

El `__init__.py` de `handlers/` NO debe importar todos los submódulos eagerly. Esto fuerza
la carga de plotly, scipy, numba, psycopg2, redis, influxdb etc. al hacer `import binpan`.
Usar un patrón lazy o simplemente dejarlo vacío y hacer imports directos donde se necesiten.

```python
# MAL - handlers/__init__.py carga todo
from . import aggregations
from . import plotting      # carga plotly
from . import postgresql     # carga psycopg2
from . import redis_fetch    # carga redis
from . import numba_tools    # carga numba
# ...20 módulos más

# BIEN - handlers/__init__.py vacío o mínimo
# Solo importar lo estrictamente necesario
# Los módulos se importan donde se usan:
from handlers.market import get_candles_by_time_stamps
```

#### Orden de imports (PEP 8)

```python
# 1. Standard library
from datetime import datetime
import os

# 2. Terceros (core siempre disponible)
import pandas as pd
import numpy as np

# 3. Locales
from handlers.logs import LogManager
from .auxiliar import csv_klines_setup
```

- Imports relativos dentro del mismo paquete (`from .auxiliar import ...`).
- Imports absolutos entre paquetes (`from handlers.exchange import ...`).

### Logging

- Usar la clase `LogManager` de `handlers/logs.py` (no `print()` para diagnósticos).
- Cada módulo crea su propio logger: `mi_logger = LogManager(filename='./logs/mi_modulo.log', name='mi_modulo', info_level='INFO')`.
- Los logs van al directorio `logs/` (gitignored).

### Excepciones

- Excepciones personalizadas en `handlers/exceptions.py`.
- Clase base: `BinPanException`. Heredar de ella para nuevas excepciones.

---

## Documentación Sphinx

- **Tema**: `shibuya`
- **Extensiones**: `sphinx.ext.autodoc`, `sphinx.ext.autosummary`, `autodocsumm`
- **Orden autodoc**: `bysource` (las funciones aparecen en el orden del código fuente)
- **Config**: `docs/conf.py`
- **Build**: `docs/_build/` (gitignored)
- Al añadir un nuevo módulo, crear su `.rst` en `docs/` y añadirlo al `toctree` de `docs/index.rst`.

---

## Seguridad y secretos

- **NUNCA commitear** claves API, secrets o credenciales.
- `secret.py` contiene claves API encriptadas y está en `.gitignore`.
- El patrón `secret*`, `*keys*` está en `.gitignore`.
- Las claves se gestionan con `pycryptodome` y se piden al usuario en primera ejecución.
- Al pushear a GitHub, verificar que no se filtren datos sensibles en el historial (squash merge).

---

## Dependencias principales

### Core (siempre necesarias)

| Paquete | Uso | Import |
|---------|-----|--------|
| `pandas` | DataFrames de velas y trades | Top-level |
| `numpy` | Cálculos numéricos | Top-level |
| `panzer` | Cliente REST Binance API (rate limiting, multi-market) | Top-level |
| `kline-timestamp` | Timestamps de velas (open/close, navegación, timezone) | Top-level |
| `requests` | Llamadas HTTP (solo para endpoints firmados; panzer lo trae como dep) | Top-level |
| `pytz` | Zonas horarias | Top-level |
| `pycryptodome` | Encriptación de claves API | Top-level |

### Opcionales (import lazy, dentro de funciones)

| Paquete | Uso | Import |
|---------|-----|--------|
| `plotly` + `kaleido` | Plotting interactivo | Lazy (solo en funciones de plot) |
| `numba` | Aceleración de indicadores (EMA, SMA, RSI, RMA) | Lazy |
| `scipy` / `scikit-learn` | Estadística y clustering | Lazy |
| `psycopg2-binary` | Conector PostgreSQL | Lazy |
| `redis` | Cache Redis | Lazy |
| `influxdb-client` | InfluxDB | Lazy |

### Restricciones de versiones (comprobado marzo 2026)

| Dependencia | Pin | Motivo del pin |
|-------------|-----|----------------|
| `numpy` | `>=1.26.2` | Desclaveado: pandas_ta eliminado, numpy 2.x compatible |
| `pandas` | `>=2.1,<3` | pandas 3.0 elimina `errors='ignore'`, `downcast`, y es estricto con tz-naive/aware |
| `setuptools` | `>=69.0.2` | Desclaveado: pandas_ta eliminado, setuptools 78+ compatible |
| `kaleido` | `>=0.2.1,<0.3` | kaleido 0.3+ rompe exportación de plotly |
| `plotly` | `>=5.18` | Plotly 6.x funciona bien con kaleido 0.2.x |

**`pandas_ta` eliminado** (marzo 2026): era una librería abandonada (última release 2021).
Todos los indicadores técnicos están ahora implementados de forma nativa en
`handlers/indicators.py`, usando `handlers/numba_tools.py` (ema_numba, rma_numba, sma_numba,
rsi_numba) como base de cálculo. Los nombres de columna son compatibles con los que usaba
pandas_ta para mantener compatibilidad con el sistema de plots.

**Paquete `typing` eliminado**: era innecesario en Python 3.10+ y causaba conflictos.

### Versiones actuales verificadas (funcionando)

```
numpy==1.26.4  pandas==2.3.3  plotly==6.6.0  scipy==1.17.1
scikit-learn==1.8.0  numba==0.64.0  requests==2.32.5  pycryptodome==3.23.0
```

### Futuras actualizaciones

- ~~**pandas_ta eliminado**~~ (completado marzo 2026). Indicadores nativos implementados.
- **Evaluar migración de `pytz` a `zoneinfo`** (stdlib desde Python 3.9).
- Al actualizar, verificar con: `import binpan; s = binpan.Symbol('BTCUSDT', '15m', limit=50); s.ema(21); s.plot()`

---

## Testing

- No hay suite de tests formal actualmente. Los notebooks `.ipynb` funcionan como pruebas manuales.
- Archivos de test están gitignored (`test*` en `.gitignore`).

---

## Notas para el desarrollo

- `symbol_manager.py` es el archivo más grande (~215K). Evaluar refactorización progresiva.
- Los Jupyter Notebooks de investigación se gitignorean (`investigacion*`, `pruebas*`).
- El entorno virtual está en `venv/` (gitignored).
- IDE: PyCharm (`.idea/` gitignored).

---

## Simplificación: delegación a librerías propias (marzo 2026)

**Rama `simp` DEPRECADA**. La simplificación se aplica directamente en `dev`.

BinPan delega funcionalidad de bajo nivel a dos librerías del mismo autor (nand0san),
publicadas en PyPI, para reducir complejidad y código duplicado.

### `panzer` — cliente REST para Binance API

- **PyPI**: `pip install panzer`
- **Requiere**: Python >=3.11, `requests`
- **Clase principal**: `BinancePublicClient(market, safety_ratio, timeout)`
- **Markets**: `"spot"`, `"um"` (USDT-M futures), `"cm"` (COIN-M futures)
- **Endpoints**: `klines()`, `trades()`, `agg_trades()`, `depth()`, `exchange_info()`,
  `ping()`, `server_time()`, `get()` (genérico)
- **Rate limiting automático**: fixed-window sincronizado con header `X-MBX-USED-WEIGHT-1M`
- **Clock sync**: `ensure_time_offset_ready()`, `now_server_ms()`

**Qué reemplaza en binpan**:

| Módulo binpan | Acción |
|---------------|--------|
| `handlers/quest.py` (endpoints públicos) | **Eliminar**: GET público, rate limiting, weight tracking → `panzer` |
| `handlers/quest.py` (endpoints firmados) | **Mantener**: POST/DELETE con API key/secret (panzer no tiene auth) |
| `handlers/market.py` (llamadas API) | **Simplificar**: `api_raw_get` → `client.klines()`, etc. Mantener transformación a DataFrame |
| `handlers/exchange.py` (llamadas API) | **Simplificar**: `api_raw_get` → `client.exchange_info()`. Mantener lógica de parseo |
| `handlers/starters.py` (rate limits) | **Simplificar**: `get_exchange_limits()` ya no es necesario |
| `objects/api.py` | **Eliminar**: ApiClient básico redundante con panzer |

### `kline-timestamp` — timestamps de velas

- **PyPI**: `pip install kline-timestamp`
- **Requiere**: Python >=3.9, `pandas>=2.3.3,<3.0`, `pytz>=2025.2`
- **Clase principal**: `KlineTimestamp(timestamp_ms: int, interval: str, tzinfo: str)`
- **Inmutable**: cada transformación devuelve nueva instancia
- **Properties**: `open`, `close`, `tick_ms`
- **Métodos**: `next()`, `prev()`, `with_timezone()`, `with_interval()`,
  `to_datetime()`, `to_pandas_timestamp()`
- **Comparadores**: `==`, `<`, `>`, `<=`, `>=` (por `(open, tick_ms)`)
- **Aritmética**: `+` / `-` con `timedelta` o entre instancias
- **Intervalos**: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w

**Qué reemplaza en binpan**:

| Módulo binpan | Acción |
|---------------|--------|
| `objects/timestamps.py` (~520 líneas) | **Eliminar**: `Timestamp` → `KlineTimestamp` directo |
| `objects/timeframes.py` | **Simplificar**: usar `KlineTimestamp` como tipo interno en vez de `Timestamp` propio |
| `handlers/time_helper.py` (funciones kline) | **Reducir**: `open_from_milliseconds` → `KlineTimestamp(ms, tick, tz).open`, etc. |
| Dicts `tick_seconds` / `tick_milliseconds` duplicados | **Eliminar**: usar `KlineTimestamp(...).tick_ms` |

**Mapeo de funciones a eliminar/reemplazar**:

```python
# ANTES (binpan propio)
from objects.timestamps import Timestamp
ts = Timestamp(ms, "Europe/Madrid", "15m")
ts.open        # int ms
ts.close       # int ms
ts.get_next_open()
ts.get_prev_close()
ts.to_datetime()

# DESPUÉS (kline-timestamp)
from kline_timestamp import KlineTimestamp
kt = KlineTimestamp(ms, "15m", "Europe/Madrid")
kt.open        # int ms
kt.close       # int ms
kt.next().open
kt.prev().close
kt.to_datetime()
```

**Lo que NO cubre `kline-timestamp`** (se mantiene en binpan):

- Parseo de strings a timestamp (`parse_timestamp()` de `objects/timestamps.py`).
  `KlineTimestamp` solo acepta `int` ms → mover a util o a `time_helper.py`.
- Conversiones genéricas datetime↔string↔ms (no específicas de klines).
- Clase `Timeframe` (rango de velas) — simplificar, no eliminar.
- Funciones pandas (inferir frecuencia, crear índices, etc.).

### Módulos que NO se tocan (valor propio de binpan)

- `binpan/symbol_manager.py` — clase `Symbol`, interfaz principal
- `handlers/indicators.py` — indicadores técnicos nativos + numba
- `handlers/numba_tools.py` — kernels numba (ema, rma, sma, rsi)
- `handlers/plotting.py` — charts con Plotly
- `handlers/tags.py` — backtesting/etiquetado
- `handlers/aggregations.py` — resampleo de klines
- `handlers/standards.py` — mapeos de columnas
- `handlers/files.py` — lectura/escritura de archivos
- `handlers/logs.py` — LogManager
- `handlers/exceptions.py` — excepciones personalizadas

---

## Plan de ejecución priorizado

La simplificación se ejecuta en este orden. Cada fase es un commit independiente.

### Fase 1: Instalar dependencias y preparar base

1. Añadir `panzer` y `kline-timestamp` a `requirements.txt` y `setup.py`.
2. Verificar compatibilidad de versiones con el entorno actual.

### Fase 2: Eliminar `objects/timestamps.py` → `kline-timestamp`

**Prioridad máxima**: es la base de la que dependen `Timeframe` y `time_helper`.

1. Identificar todos los usos de `Timestamp` en el código (`import`, instanciación, atributos).
2. Reemplazar por `KlineTimestamp` en cada punto de uso.
3. Extraer `parse_timestamp()` a `handlers/time_helper.py` (es la única función que no
   cubre `kline-timestamp`, ya que este solo acepta `int` ms).
4. Eliminar `objects/timestamps.py`.
5. Verificar: importar binpan y ejecutar operaciones con timestamps.

### Fase 3: Simplificar `objects/timeframes.py`

**Depende de fase 2**.

1. Refactorizar `Timeframe` para usar `KlineTimestamp` en vez de `Timestamp`.
2. Eliminar dict `tick_milliseconds` duplicado (usar `KlineTimestamp(...).tick_ms`).
3. Simplificar métodos que reimplementaban lógica de open/close.
4. Verificar: iterar timeframes, `len()`, `__contains__`, indexing.

### Fase 4: Reducir `handlers/time_helper.py`

**Depende de fase 2 y 3**.

1. Eliminar funciones de kline boundaries ya cubiertas por `KlineTimestamp`:
   `open_from_milliseconds`, `next_open_by_milliseconds`, `close_from_milliseconds`,
   `next_open_utc`.
2. Eliminar dict `tick_seconds` duplicado.
3. Mantener: conversiones genéricas (string↔datetime↔ms), `parse_timestamp()`,
   funciones pandas, `calculate_iterations`, `check_tick_interval`.
4. Actualizar callers en `market.py` y `symbol_manager.py`.

### Fase 5: Integrar `panzer` en `handlers/market.py`

**Independiente de fases 2-4** (se puede paralelizar).

1. Crear instancia compartida de `BinancePublicClient` (lazy, al primer uso).
2. Reemplazar llamadas `api_raw_get()` por métodos de panzer:
   - `get_candles_api()` → `client.klines()`
   - `get_agg_trades_api()` → `client.agg_trades()`
   - `get_last_price()` → `client.get("/api/v3/ticker/price", ...)`
3. Mantener toda la lógica de transformación API response → DataFrame.
4. Eliminar imports de `quest.py` para endpoints públicos.

### Fase 6: Integrar `panzer` en `handlers/exchange.py`

**Depende de fase 5** (comparten el cliente panzer).

1. Reemplazar `api_raw_get()` → `client.exchange_info()`, `client.get()`.
2. Mantener parseo de exchange info (filtros, símbolos, etc.).
3. Eliminar imports de `quest.py` para endpoints públicos.

### Fase 7: Reducir `handlers/quest.py`

**Depende de fases 5 y 6**.

1. Eliminar: `get_response()`, `check_weight()`, `update_weights()`, `get_server_time()`,
   `api_raw_get()`, toda la maquinaria de rate limiting para endpoints públicos.
2. Mantener: `sign_request()`, `get_signed_request()`, `post_signed_request()`,
   `delete_signed_request()`, `hashed_signature()` (endpoints con autenticación).
3. Evaluar si renombrar el módulo (`quest.py` → `auth.py` o similar).

### Fase 8: Eliminar `objects/api.py`

**Tras fase 5**. Es redundante con panzer. Eliminar y actualizar imports.

### Fase 9: Limpieza final

1. Migrar `typing` deprecado → builtins en archivos tocados (aprovechar cada fase).
2. Eliminar star imports pendientes.
3. Actualizar `handlers/__init__.py` (lazy o vacío).
4. Actualizar `requirements.txt` y `setup.py` (quitar `requests` si ya no se usa directamente
   en endpoints públicos; panzer lo trae como dependencia).
5. Actualizar documentación Sphinx (`.rst` de módulos eliminados/renombrados).
6. Test completo: `import binpan; s = binpan.Symbol('BTCUSDT', '15m', limit=50); s.ema(21); s.plot()`

---

## Deuda técnica conocida - Imports (estado actual)

### Completado (marzo 2026)

- **`typing` deprecado → builtins**: Migrado en todos los archivos. Solo queda `from typing import Literal` en `plotting.py` (correcto, no tiene equivalente builtin).
- **Star imports eliminados**: `binpan/__init__.py` y `handlers/postgresql.py` ahora usan imports explícitos.
- **`handlers/__init__.py`**: Ya es lazy via `__getattr__` + `importlib`.

### Pendiente

| Archivo | Import con fallback |
|---------|-------------------|
| `handlers/redis_fetch.py` | `from redis import StrictRedis` (try/except) |
| `handlers/influx_manager.py` | `import influxdb_client` (try/except) |
| `handlers/starters.py` | `importlib.import_module('secret')` |
