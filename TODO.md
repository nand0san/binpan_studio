# TODO - BinPan Studio

Próximos pasos de desarrollo, ordenados por prioridad.

---

## ~~1. Correctitud e imports~~ (COMPLETADO)

- [x] `handlers/redis_fetch.py`: import directo de `StrictRedis` (sin try/except)
- [x] `handlers/influx_manager.py`: import directo de `influxdb_client` (sin try/except)
- [x] `handlers/starters.py`: `import_secret_module()` lanza `ModuleNotFoundError` con mensaje claro
- [x] `handlers/wallet.py`: eliminado `convert_str_date_to_ms_old()` duplicado, corregido import mixto

---

## ~~2. Refactorización de symbol_manager.py~~ (COMPLETADO)

`symbol_manager.py` reducido de **4.192 → 1.449 líneas** (-65%) usando patrón mixin:

| Mixin | Archivo | Líneas | Métodos |
|-------|---------|--------|---------|
| `IndicatorsMixin` | `binpan/indicators_mixin.py` | 1.389 | 26 (ema, rsi, macd, bbands, supertrend, etc.) |
| `PlottingMixin` | `binpan/plotting_mixin.py` | 755 | 20 (plot, set_plot_*, plot_trades_*, etc.) |
| `StrategyMixin` | `binpan/strategy_mixin.py` | 668 | 12 (backtesting, tag, cross, roi, etc.) |

`class Symbol(IndicatorsMixin, PlottingMixin, StrategyMixin):` - API pública intacta.

---

## ~~3. Bugs y warnings~~ (COMPLETADO)

### Bugs corregidos

- [x] `check_weight` NameError: función eliminada del import pero 3 llamadas huérfanas en `market.py` (×2) y `exchange.py` (×1)
- [x] `from_csv` AttributeError: `self.tick_interval` no inicializado antes de `csv_klines_setup()` en `symbol_manager.py`

### Deprecation warnings eliminados

- [x] Pandas freq aliases: `'T'` → `'min'`, `'H'` → `'h'`, `'M'` → `'ME'` en `time_helper.py` y `timeframes.py`
- [x] `datetime.utcfromtimestamp()` → `datetime.fromtimestamp(ts, tz=timezone.utc)` en `time_helper.py` (6), `exchange.py`, `timeframes.py`, `influx_manager.py`
- [x] `datetime.utcnow()` → `datetime.now(timezone.utc)` en `influx_manager.py` (4)
- [x] Plotly `fig.append_trace()` → `fig.add_trace()` en `plotting.py`

### Warning externo (no nuestro, no corregible)

- [ ] kaleido 0.2.x + plotly 6.x: `DeprecationWarning: Kaleido < 1.0.0`. Pinned en setup.py (`kaleido>=0.2.1,<0.3`) porque kaleido 0.3+ rompe la exportación de imágenes.

---

## ~~4. Verificación notebooks~~ (COMPLETADO)

Todos los notebooks ejecutados y verificados:

| Notebook | Estado | Notas |
|---|---|---|
| `basic tutorial` | OK | klines, indicadores, plot, exchange data |
| `examples analysis` | OK | atomic trades, S/R, orderbook |
| `examples technical indicators` | OK | 26 indicadores, fractals, market profile |
| `examples plotting module` | OK | 22 tests de plotting |
| `examples tagging & backtesting` | OK | cross, tag, backtesting, roi |
| `Ichimoku Analysis/Backtesting` | OK | ichimoku + cross + backtesting |
| `examples Reversal Chart` | OK | reversal candles desde agg trades |
| `examples S/R KMEANS` | OK | S/R simple + rolling |
| `export CSV data` | OK | save_csv + from_csv |
| `examples exchange` | SKIP | requiere API key (diseño) |
| `examples database*` | SKIP | requiere PostgreSQL/Redis/InfluxDB |

### Bugs en notebooks (no en el código)

Algunos notebooks usan API obsoleta. Corregir cuando se actualicen:

- `examples plotting`: `markers=True` (debe ser lista), `candles_ta(time_zone=...)` y `candles_ta(hlines=...)` (parámetros inexistentes), `plot_hists_vs(name0=...)` (debe ser `x0_name`)
- `examples analysis`: `FIROUSDC` delistado de Binance

---

## ~~5. Limpieza de código muerto~~ (COMPLETADO)

Eliminadas **~371 líneas** de código muerto tras integración de panzer y kline-timestamp:

| Archivo | Eliminado | Detalle |
|---------|-----------|---------|
| `handlers/time_helper.py` | 230 líneas | 19 funciones sin uso externo (wrappers redundantes con KlineTimestamp, utilidades de tiempo sin callers) |
| `handlers/starters.py` | 82 líneas | `get_exchange_limits()`, `is_python_version_numba_supported()`, `is_running_in_jupyter()` + import `requests` |
| `handlers/quest.py` | 10 líneas | `tick_seconds` duplicado, parámetro `weight` sin uso (check_weight eliminado) |
| `handlers/exchange.py` | 4 líneas | `float_api_items`, `int_api_items` duplicados (solo usados en quest.py) |
| `binpan/auxiliar.py` | 45 líneas | bloque comentado `setup_startime_endtime` (referenciaba Timestamp eliminada) |

---

## 6. Limpieza residual (prioridad alta, esfuerzo bajo)

Hallazgos de la auditoría de marzo 2026:

- [ ] `calculate_iterations()` en `time_helper.py`: dead code (definida pero no importada ni usada en ningún módulo)
- [ ] `pandas_freq_tick_interval` duplicado: definido en `time_helper.py` y `objects/timeframes.py`. `timeframes.py` debería importarlo de `time_helper.py`

---

## 7. Documentación Sphinx (prioridad media)

### 7.1 RST faltantes para módulos nuevos

Crear archivos `.rst` y añadir al `toctree` de `docs/index.rst`:

- [ ] `docs/indicators_mixin.rst` → `binpan/indicators_mixin.py`
- [ ] `docs/plotting_mixin.rst` → `binpan/plotting_mixin.py`
- [ ] `docs/strategy_mixin.rst` → `binpan/strategy_mixin.py`
- [ ] `docs/timeframes.rst` → `objects/timeframes.py` (clase Timeframe)
- [ ] `docs/trades_object.rst` → `objects/trades.py` (clase Trades)
- [ ] `docs/numba_tools.rst` → `handlers/numba_tools.py`
- [ ] `docs/logs.rst` → `handlers/logs.py` (clase LogManager)

### 7.2 RST a limpiar (módulos eliminados)

- [ ] Verificar que no haya referencias a `objects/timestamps.py` o `objects/api.py` en docs existentes
- [ ] Actualizar cualquier referencia a la API antigua (`.get_open()` → `.open`, `.timezone` → `.timezone_IANA`)

### 7.3 Actualizar notebooks de ejemplo

- [ ] Corregir los bugs de notebooks listados en sección 4

---

## 8. Tests (prioridad media)

No hay suite de tests formal. Los notebooks sirven como tests manuales pero no son automatizables.

### 8.1 Test suite básica con pytest

```
tests/
├── test_symbol.py          # Symbol básico: crear, get_timestamps, get_dates
├── test_indicators.py      # EMA, RSI, MACD, BBands con datos mock
├── test_timeframe.py       # Timeframe: iteración, start/end, timezone
├── test_time_helper.py     # open_from_milliseconds, etc.
├── test_market.py          # get_last_price, get_candles (con mock de panzer)
└── conftest.py             # Fixtures: DataFrames de ejemplo, mock de API
```

**Priorizar**: tests de `time_helper.py` y `timeframes.py` (lógica pura, sin API).

---

## 9. Modernización (prioridad baja)

### 9.1 Migrar pytz → zoneinfo

3 archivos usan `pytz` (`time_helper.py`, `indicators.py`, `timeframes.py`).
`zoneinfo` es stdlib desde Python 3.9 y el proyecto requiere 3.10+.

**Precaución**: pytz y zoneinfo manejan DST de forma diferente. Testear bien antes de migrar.

### 9.2 Intervalo "1M" (mensual)

`kline-timestamp` no soporta "1M". Si se necesita, abrir issue o implementar fallback
específico para este caso en `time_helper.py`.

### 9.3 Evaluar eliminación de dependencia `requests`

`panzer` ya trae `requests` como dependencia transitiva. Evaluar si se puede quitar
del `requirements.txt` directo (solo necesaria en `quest.py` para requests autenticadas).

---

## 10. TODOs en el código

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
| ~~1~~ | ~~Eliminar fallbacks silenciosos~~ | ~~Bajo~~ | ~~Alto (correctitud)~~ |
| ~~2~~ | ~~Refactorizar symbol_manager.py~~ | ~~Alto~~ | ~~Alto (mantenibilidad)~~ |
| ~~3~~ | ~~Bugs y warnings~~ | ~~Bajo~~ | ~~Alto (correctitud)~~ |
| ~~4~~ | ~~Verificación notebooks~~ | ~~Medio~~ | ~~Alto (validación)~~ |
| ~~5~~ | ~~Limpieza código muerto~~ | ~~Medio~~ | ~~Medio (mantenibilidad)~~ |
| 6 | Limpieza residual (auditoría) | Bajo | Medio (mantenibilidad) |
| 7 | Documentar módulos en Sphinx | Medio | Medio (documentación) |
| 8 | Crear test suite pytest | Medio | Alto (calidad) |
| 9 | Migrar pytz → zoneinfo | Bajo | Bajo (modernización) |
| 10 | Resolver TODOs del código | Variable | Variable |

---

## Estado del proyecto (auditoría marzo 2026)

**~19.800 líneas** de código Python (5.154 binpan + 14.302 handlers + 375 objects).

| Categoría | Estado |
|-----------|--------|
| Imports (typing, star, fallbacks) | ✅ Correcto |
| Integración panzer / kline-timestamp | ✅ Completada |
| Deprecation warnings | ✅ Eliminados (excepto kaleido externo) |
| `__init__.py` lazy loading | ✅ Correcto (binpan + handlers) |
| Versión sincronizada (setup.py / symbol_manager.py) | ✅ `0.8.14` |
| Tests formales | ❌ No existen |
| Documentación Sphinx | ⚠️ 7 módulos sin .rst |
| pytz (3 archivos) | ⚠️ Pendiente migración a zoneinfo |
