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
│   ├── exchange.py         # Llamadas a API de exchange
│   ├── market.py           # Llamadas a API de mercado (velas, trades)
│   ├── indicators.py       # Indicadores técnicos propios
│   ├── plotting.py         # Plots con Plotly
│   ├── tags.py             # Etiquetado y backtesting
│   ├── time_helper.py      # Utilidades de tiempo
│   ├── files.py            # Lectura/escritura de archivos
│   ├── quest.py            # Peticiones HTTP a API
│   ├── standards.py        # Constantes y mapeos de columnas
│   ├── logs.py             # LogManager (logging rotativo)
│   ├── exceptions.py       # Excepciones personalizadas
│   ├── aggregations.py     # Resampleo de klines
│   ├── postgresql.py       # Conector PostgreSQL
│   ├── numba_tools.py      # Indicadores acelerados con Numba
│   └── ...
├── objects/             # Nuevas clases de dominio (en desarrollo)
│   ├── timestamps.py       # Clase Timestamp
│   ├── timeframes.py       # Clase Timeframe
│   ├── trades.py           # Clase Trades (hereda de Timeframe)
│   └── api.py              # ApiClient básico
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
| `simp` | Rama experimental (simplificación). |

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
| `requests` | Llamadas HTTP a API de Binance | Top-level |
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
- El directorio `objects/` contiene el nuevo diseño (Timestamp, Timeframe, Trades) que busca
  modularizar la lógica. Está en desarrollo activo.
- Los Jupyter Notebooks de investigación se gitignorean (`investigacion*`, `pruebas*`).
- El entorno virtual está en `venv/` (gitignored).
- IDE: PyCharm (`.idea/` gitignored).

---

## Deuda técnica conocida - Imports (estado actual)

Inventario de lo que hay que migrar progresivamente:

### Fallbacks a eliminar (hacer import directo, sin try/except)

| Archivo | Import con fallback |
|---------|-------------------|
| `handlers/redis_fetch.py` | `from redis import StrictRedis` |
| `handlers/influx_manager.py` | `import influxdb_client` |
| `handlers/starters.py` | `importlib.import_module('secret')` |

### `typing` deprecado a migrar → builtins

18 archivos usan `from typing import Tuple, List, Dict, Union, Optional`.
Migrar a `tuple`, `list`, `dict`, `X | Y`, `X | None`.

Archivos afectados: `symbol_manager.py`, `auxiliar.py`, `aggregations.py`, `exchange.py`,
`indicators.py`, `market.py`, `numba_tools.py`, `postgresql.py`, `quest.py`, `redis_fetch.py`,
`tags.py`, `time_helper.py`, `influx_manager.py`, `plotting.py`, `stat_tests.py`,
`database_connector.py`, `objects/trades.py`, `objects/timeframes.py`, `objects/timestamps.py`.

### Star imports a eliminar

| Archivo | Import |
|---------|--------|
| `binpan/__init__.py` | `from .auxiliar import *` |
| `handlers/postgresql.py` | `from .standards import *` |

### `handlers/__init__.py` - carga eager de 20 módulos

Actualmente importa todos los submódulos al cargar el paquete. Vaciar o hacer lazy.
