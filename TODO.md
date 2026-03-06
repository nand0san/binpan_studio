# TODO - BinPan Studio

Tareas pendientes, ordenadas por prioridad.
Versión actual: **0.9.0** · ~19.400 líneas en 38 archivos Python.

---

## 1. Tests pytest (prioridad alta)

No hay suite de tests formal. Los notebooks sirven como tests manuales pero no son automatizables.

### Diseño propuesto

```
tests/
├── conftest.py             # Fixtures: DataFrames de ejemplo, mock de panzer
├── test_symbol.py          # Symbol básico: crear, get_timestamps, get_dates
├── test_indicators.py      # EMA, RSI, MACD, BBands con datos mock
├── test_timeframe.py       # Timeframe: iteración, start/end, timezone
├── test_time_helper.py     # parse_timestamp, conversiones
├── test_market.py          # get_last_price, get_candles (con mock de panzer)
├── test_aggregations.py    # Resampleo de klines
└── DATA_PROPERTIES.md      # Referencia de invariantes de datos (ya existe)
```

**Decisión de diseño**: las invariantes de datos crudos de la API (unicidad IDs, secuencialidad,
OHLCV constraints) se testean en **panzer**, no en BinPan.
BinPan solo testea su lógica propia: transformación a DataFrame, índices, indicadores, plots.

**Priorizar**: tests de `time_helper.py`, `timeframes.py` e `indicators.py` (lógica pura, sin API).

---

## 2. Notebooks (prioridad media)

Imports corregidos, métodos faltantes añadidos, bugs de plotting arreglados.

**Ejecutado y verificado (0 errores):**
- `04_plotting.ipynb` — 68 celdas de código, todos los plots generados correctamente

**Pendiente re-ejecutar para verificar:**
- `01_basic_tutorial.ipynb`
- `02_data_analysis.ipynb`
- `03_technical_indicators.ipynb`
- `06_tagging_and_backtesting.ipynb`
- `07_support_resistance_kmeans.ipynb`
- `11_exchange_info.ipynb`
- `13_timescale_backend.ipynb`
- `15_database_maintenance.ipynb`

No demostrado: `plot_orderbook_value` (requiere datos de streaming en formato especial).

---

## 3. Modernización (prioridad baja)

### 3.1 Migrar pytz → zoneinfo

3 archivos usan `pytz` (`time_helper.py`, `indicators.py`, `timeframes.py`).
`zoneinfo` es stdlib desde Python 3.9 y el proyecto requiere 3.12+.

**Precaución**: pytz y zoneinfo manejan DST de forma diferente. Testear bien antes de migrar.
**Nota**: `kline-timestamp` depende de `pytz` — verificar si también se actualiza.

### 3.2 Warning externo kaleido

kaleido 0.2.x + plotly 6.x: `DeprecationWarning: Kaleido < 1.0.0`. Pinned en setup.py
(`kaleido>=0.2.1,<0.3`) porque kaleido 0.3+ rompe la exportación de imágenes.
Monitorizar releases de kaleido 1.0+ para evaluar actualización.

---

## 4. TODOs en el código

| Archivo | Nota | Esfuerzo |
|---------|------|----------|
| `binpan/storage/redis_fetch.py:762` | "actualizar con lo nuevo de binance cache" | Alto (diseño) |
| `binpan/api/wallet_api.py:961,986` | "saber si el interest tiene signo negativo" (×2) | Medio (investigar API margin) |
| `binpan/core/trades.py:63` | "IMPLEMENTAR FUNCIONES PARA PARSEO DE DISTINTOS ORIGENES DE DATOS" | Alto (diseño) |

---

## Resumen

| # | Tarea | Esfuerzo | Impacto |
|---|-------|----------|---------|
| 1 | Tests pytest | Medio | Alto |
| 2 | Re-ejecutar notebooks (7 pendientes) | Bajo | Medio |
| 3 | Modernización (pytz, kaleido) | Bajo | Bajo |
| 4 | TODOs en el código | Variable | Variable |

---

## Estado del proyecto (marzo 2026)

| Categoría | Estado |
|-----------|--------|
| Versión | `0.9.0` (setup.py + symbol.py) |
| Python requerido | `>=3.12.0` |
| Imports (typing, star, fallbacks) | OK |
| Integración panzer 2.2.0 (público + auth + parallel_get/bulk) | OK |
| Integración kline-timestamp 0.3.0 (incl. 1M mensual) | OK |
| Credenciales Binance | OK (`~/.panzer_creds` vía panzer) |
| Credenciales Telegram/PostgreSQL | OK (`secret.py` + `AesCipher`) |
| Deprecation warnings | OK (excepto kaleido externo) |
| Lazy loading | OK (binpan + subpaquetes) |
| Estructura de paquetes | OK (api/, core/, analysis/, plotting/, storage/) |
| Refactorización Symbol | OK (symbol.py + 3 mixins: indicators, plotting, strategy) |
| Documentación Sphinx | OK (22 RST actualizados + 5 nuevos) |
| Notebooks | OK (imports y plots corregidos, 04_plotting verificado) |
| Colores taker/maker | OK (normalizados: buyer=verde, seller=rojo) |
| Títulos S/R en plots | OK (indican fuente y calidad de datos) |
| Tests formales | No existen |
| pytz (3 archivos) | Pendiente migración a zoneinfo |
