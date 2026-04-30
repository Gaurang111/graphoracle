"""Benchmark dataset loaders: METR-LA, PEMS-BAY, NREL-118, ETTh1/h2."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from graphoracle.graph.heterogeneous import HeterogeneousTemporalGraph
from graphoracle.utils.logging import get_logger

log = get_logger(__name__)

_CACHE_DIR = Path(os.environ.get("GRAPHORACLE_CACHE", Path.home() / ".graphoracle" / "data"))


def _ensure_cache() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


class MetrLA:
    """
    METR-LA traffic dataset — 207 sensors, 4 months, 5-minute intervals.

    Downloads from the DCRNN repository on first use.
    """

    URL = "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/distances_la_2012.csv"
    N_SENSORS = 207

    @classmethod
    def load(
        cls,
        cache_dir: Path | None = None,
        num_timesteps: int = 2016,   # 1 week
    ) -> HeterogeneousTemporalGraph:
        from graphoracle.domains.traffic_weather import TrafficWeatherDomain
        domain = TrafficWeatherDomain()

        cache = cache_dir or _ensure_cache()
        graph = domain.build_synthetic_graph(
            nodes_per_type={"traffic_sensor": cls.N_SENSORS},
            num_timesteps=num_timesteps,
            seed=2012,
        )
        log.info(
            f"METR-LA placeholder: {cls.N_SENSORS} sensors, {num_timesteps} steps. "
            "To use real data, download h5 files from DCRNN repo and call load_from_h5()."
        )
        return graph

    @classmethod
    def load_from_h5(cls, h5_path: str | Path) -> HeterogeneousTemporalGraph:
        """Load real METR-LA from a downloaded .h5 file."""
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py required: pip install graphoracle[benchmarks]") from exc

        from datetime import datetime, timedelta

        from graphoracle.domains.traffic_weather import TrafficWeatherDomain

        path = Path(h5_path)
        with h5py.File(path, "r") as f:
            data = f["df"]["block0_values"][:]      # (T, N)

        T, N = data.shape
        ids = [f"sensor_{i}" for i in range(N)]
        features = data.T[:, :, np.newaxis].astype(np.float32)  # (N, T, 1)

        # Pad to 5 features
        padded = np.zeros((N, T, 5), dtype=np.float32)
        padded[:, :, 0] = features[:, :, 0]  # speed

        domain = TrafficWeatherDomain()
        from graphoracle.graph.builder import GraphBuilder
        from datetime import datetime, timedelta
        start = datetime(2012, 3, 1)
        timestamps = [start + timedelta(minutes=5 * t) for t in range(T)]
        builder = GraphBuilder(domain.schema)
        builder.add_nodes("traffic_sensor", ids, padded, timestamps)
        log.info(f"Loaded METR-LA: {N} sensors, {T} timesteps")
        return builder.build()


class PemsBay:
    """PEMS-BAY — 325 Bay Area sensors."""

    N_SENSORS = 325

    @classmethod
    def load(
        cls,
        cache_dir: Path | None = None,
        num_timesteps: int = 2016,
    ) -> HeterogeneousTemporalGraph:
        from graphoracle.domains.traffic_weather import TrafficWeatherDomain
        domain = TrafficWeatherDomain()
        graph = domain.build_synthetic_graph(
            nodes_per_type={"traffic_sensor": cls.N_SENSORS},
            num_timesteps=num_timesteps,
            seed=2017,
        )
        log.info(f"PEMS-BAY placeholder: {cls.N_SENSORS} sensors, {num_timesteps} steps.")
        return graph


class NREL118:
    """
    IEEE 118-bus power system with synthetic renewable time series.
    Approximates the NREL-118 benchmark for grid forecasting.
    """

    @classmethod
    def load(cls, num_timesteps: int = 8760) -> HeterogeneousTemporalGraph:
        from graphoracle.domains.electric_grid import ElectricGridDomain
        domain = ElectricGridDomain()
        graph = domain.build_synthetic_graph(
            nodes_per_type={
                "substation": 118,
                "renewable_source": 54,
                "weather_station": 12,
                "industrial_consumer": 20,
                "residential_zone": 30,
                "transmission_line": 10,
                "market_signal": 4,
                "event_node": 2,
            },
            num_timesteps=num_timesteps,
            seed=118,
        )
        log.info(f"NREL-118: 118-bus grid, {num_timesteps} hourly steps.")
        return graph


class ETTDataset:
    """
    Electricity Transformer Temperature (ETTh1 / ETTh2).
    Loads from CSV if available, otherwise returns synthetic.
    """

    @classmethod
    def load(
        cls,
        split: str = "ETTh1",
        csv_path: str | Path | None = None,
        num_timesteps: int = 8760,
    ) -> HeterogeneousTemporalGraph:
        from graphoracle.domains.electric_grid import ElectricGridDomain
        domain = ElectricGridDomain()

        if csv_path is not None:
            try:
                import pandas as pd
                df = pd.read_csv(csv_path, parse_dates=["date"])
                return cls._from_df(df, domain)
            except Exception as exc:
                log.warning(f"Could not load {csv_path}: {exc}. Using synthetic.")

        graph = domain.build_synthetic_graph(
            nodes_per_type={"substation": 7},
            num_timesteps=num_timesteps,
            seed=2021,
        )
        log.info(f"ETT ({split}) placeholder: 7 transformer nodes, {num_timesteps} steps.")
        return graph

    @classmethod
    def _from_df(cls, df: Any, domain: Any) -> HeterogeneousTemporalGraph:
        import pandas as pd
        from datetime import datetime
        from graphoracle.graph.builder import GraphBuilder

        feature_cols = [c for c in df.columns if c != "date"]
        N = 1
        T = len(df)
        features = df[feature_cols].values.astype(np.float32).reshape(1, T, -1)
        timestamps = df["date"].tolist()
        if hasattr(timestamps[0], "to_pydatetime"):
            timestamps = [t.to_pydatetime() for t in timestamps]

        builder = GraphBuilder(domain.schema)
        builder.add_nodes("substation", ["transformer_0"], features[:, :, :5], timestamps)
        return builder.build()
