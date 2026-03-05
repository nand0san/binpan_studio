# TODO - BinPan Studio

Próximos pasos de desarrollo, ordenados por prioridad.

---

## 1. Correctitud e imports (prioridad alta)

### 1.1 Eliminar fallbacks silenciosos en imports

Según la política de CLAUDE.md: si una dependencia no está, que dé error claro.

| Archivo | Problema | Acción |
|---------|----------|--------|
| `handlers/redis_fetch.py` | `try: from redis import StrictRedis` → `StrictRedis = None` | Import directo, sin try/except |
| `handlers/influx_manager.py` | `try: import influxdb_client` → `= None` | Import directo, sin try/except |
| `handlers/starters.py` | `importlib.import_module('secret')` con fallback silencioso | Hacer fallar explícitamente si se necesitan credenciales |

### 1.2 Eliminar código duplicado

- `handlers/wallet.py`: `convert_str_date_to_ms_old()` es duplicado de `convert_str_date_to_ms()`. Eliminar la versión `_old`.

---

## 2. Refactorización de symbol_manager.py (prioridad alta)

`binpan/symbol_manager.py` tiene **4.192 líneas** y viola el principio de responsabilidad única.
Contiene ~90 métodos que mezclan: datos OHLC, trades, indicadores, plotting y backtesting.

### Plan progresivo de extracción

| Grupo de métodos | Destino propuesto | Líneas aprox. |
|-----------------|-------------------|---------------|
| Indicadores técnicos (`ema`, `sma`, `rsi`, `macd`, `bbands`, `stoch`, `ichimoku`, etc.) | Mixin o módulo `binpan/indicators_mixin.py` | ~800 |
| Configuración de plots (`set_plot_*`, `plot`) | Mixin o módulo `binpan/plotting_mixin.py` | ~600 |
| Backtesting y tagging (`tag_*`, `backtesting`) | Mixin o módulo `binpan/backtesting_mixin.py` | ~400 |
| Gestión de trades (`get_agg_trades`, `get_atomic_trades`) | Integrar con `objects/trades.py` | ~300 |

**Estrategia**: Usar mixins para mantener la API `s.ema(21)` intacta, sin romper código existente.

---

## 3. Documentación Sphinx (prioridad media)

### 3.1 RST faltantes para módulos nuevos

Crear archivos `.rst` y añadir al `toctree` de `docs/index.rst`:

- [ ] `docs/timeframes.rst` → `objects/timeframes.py` (clase Timeframe)
- [ ] `docs/trades_object.rst` → `objects/trades.py` (clase Trades)
- [ ] `docs/numba_tools.rst` → `handlers/numba_tools.py`
- [ ] `docs/logs.rst` → `handlers/logs.py` (clase LogManager)

### 3.2 RST a limpiar (módulos eliminados)

- [ ] Verificar que no haya referencias a `objects/timestamps.py` o `objects/api.py` en docs existentes
- [ ] Actualizar cualquier referencia a la API antigua (`.get_open()` → `.open`, `.timezone` → `.timezone_IANA`)

### 3.3 Actualizar notebooks de ejemplo

Los 16 notebooks existentes pueden tener imports o patrones obsoletos tras la simplificación.
Revisar al menos `basic tutorial.ipynb` y `examples analysis.ipynb`.

---

## 4. Tests (prioridad media)

No hay suite de tests formal. Los notebooks sirven como tests manuales pero no son automatizables.

### 4.1 Test suite básica con pytest

```
tests/
├── test_symbol.py          # Symbol básico: crear, get_timestamps, get_dates
├── test_indicators.py      # EMA, RSI, MACD, BBands con datos mock
├── test_timeframe.py       # Timeframe: iteración, start/end, timezone
├── test_time_helper.py     # open_from_milliseconds, close_from_milliseconds, etc.
├── test_market.py          # get_last_price, get_candles (con mock de panzer)
└── conftest.py             # Fixtures: DataFrames de ejemplo, mock de API
```

**Priorizar**: tests de `time_helper.py` y `timeframes.py` (lógica pura, sin API).

---

## 5. Modernización (prioridad baja)

### 5.1 Migrar pytz → zoneinfo

3 archivos usan `pytz` (`time_helper.py`, `indicators.py`, `timeframes.py`).
`zoneinfo` es stdlib desde Python 3.9 y el proyecto requiere 3.10+.

**Precaución**: pytz y zoneinfo manejan DST de forma diferente. Testear bien antes de migrar.

### 5.2 Intervalo "1M" (mensual)

`kline-timestamp` no soporta "1M". Si se necesita, abrir issue o implementar fallback
específico para este caso en `time_helper.py`.

### 5.3 Evaluar eliminación de dependencia `requests`

`panzer` ya trae `requests` como dependencia transitiva. Evaluar si se puede quitar
del `requirements.txt` directo (solo necesaria en `quest.py` para requests autenticadas).

---

## 6. TODOs en el código

Comentarios `TODO` encontrados en el código fuente:

| Archivo | Línea | Nota |
|---------|-------|------|
| `handlers/indicators.py` | ~947 | "si los trades rulan dejamos esto como referencia para las klines" |
| `handlers/redis_fetch.py` | ~768 | "actualizar con lo nuevo de binance cache" |
| `handlers/wallet.py` | ~1104, ~1129 | "saber si el interest tiene signo negativo" (×2) |
| `objects/trades.py` | ~63 | "IMPLEMENTAR FUNCIONES PARA PARSEO DE DISTINTOS ORÍGENES DE DATOS" |

---

## Resumen de prioridades

| # | Tarea | Esfuerzo | Impacto |
|---|-------|----------|---------|
| 1 | Eliminar fallbacks silenciosos | Bajo | Alto (correctitud) |
| 2 | Refactorizar symbol_manager.py | Alto | Alto (mantenibilidad) |
| 3 | Documentar módulos objects/ en Sphinx | Medio | Medio (documentación) |
| 4 | Crear test suite pytest | Medio | Alto (calidad) |
| 5 | Migrar pytz → zoneinfo | Bajo | Bajo (modernización) |
| 6 | Resolver TODOs del código | Variable | Variable |
