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

Imports actualizados y métodos faltantes añadidos. Pendiente re-ejecutar para verificar.

Métodos añadidos a notebooks:
- `ma()` → 03_technical_indicators (wrapper genérico de medias móviles)
- `set_plotting_volume_ma` → 04_plotting (MA en panel de volumen)
- `dist_plot` → 04_plotting (distribución con KDE)
- `set_plot_splitted_serie_couple` → 04_plotting (área coloreada bull/bear entre EMAs)
- `ffill_window` → 06_tagging_and_backtesting (propagar señales N velas)
- `set_strategy_groups` → 06_tagging_and_backtesting (agrupar columnas de estrategia)

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
| 2 | Re-ejecutar notebooks | Bajo | Medio |
| 3 | Modernización (pytz, kaleido) · 1M resuelto | Bajo | Bajo |
| 4 | TODOs en el código | Variable | Variable |

---

## Estado del proyecto (marzo 2026)

| Categoría | Estado |
|-----------|--------|
| Versión | `0.9.0` (setup.py + symbol.py) |
| Python requerido | `>=3.12.0` |
| Imports (typing, star, fallbacks) | OK |
| Integración panzer 2.1.0 (público + auth) | OK |
| Integración kline-timestamp | OK |
| Credenciales Binance | OK (`~/.panzer_creds` vía panzer) |
| Credenciales Telegram/PostgreSQL | OK (`secret.py` + `AesCipher`) |
| Deprecation warnings | OK (excepto kaleido externo) |
| Lazy loading | OK (binpan + subpaquetes) |
| Estructura de paquetes | OK (api/, core/, analysis/, plotting/, storage/) |
| Documentación Sphinx | OK (rutas actualizadas, RST nuevos creados) |
| Notebooks | OK (imports actualizados, pendiente re-ejecutar) |
| Tests formales | No existen |
| pytz (3 archivos) | Pendiente migración a zoneinfo |
