from .symbol import Symbol, __version__
from .wallet import Wallet
from .exchange import Exchange
from .auxiliar import (csv_klines_setup, check_continuity, repair_kline_discontinuity,
                       find_common_interval_and_generate_timestamps, add_missing_klines, fill_missing_values)


def __getattr__(name):
    if name == "Database":
        from .database import Database
        globals()["Database"] = Database
        return Database
    raise AttributeError(f"module 'binpan' has no attribute {name!r}")
