# GraphOracle

**Multi-domain Heterogeneous Temporal Graph Forecasting Framework**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## What is GraphOracle?

Most real-world forecasting problems are not isolated time series. A power outage at one substation cascades through the grid. A port delay ripples across the entire supply chain. A variant mutation accelerates epidemic spread in connected regions.

**GraphOracle** is a framework for heterogeneous temporal graph forecasting. It wraps established GNN architectures — HGT, TGN, STGNN, LSTM, GRU — in a unified training loop with a domain schema system, evaluation pipeline, and uncertainty estimation. The actual forecasting is performed by these underlying models; GraphOracle provides the plumbing around them.

GraphOracle ships with 5 built-in domain schemas (electric grid, supply chain, traffic, epidemic, finance). Adding a new domain requires writing a schema class that defines node types, edge types, and features — there is no automatic type inference from raw data.

Key capabilities:

- **Heterogeneous graphs** — nodes and edges have types; models learn type-specific representations
- **Temporal modelling** — TGN-style memory, sinusoidal time encoding, irregular sampling
- **Unified training loop** — data prep, checkpointing, evaluation, and early stopping handled by the framework
- **5 built-in domains** — electric grid, supply chain, traffic, epidemic, finance; new domains require a manual schema definition (~30 lines)
- **Custom GNN models** — three levels from "swap the backbone" to "full architecture control"
- **Cascade simulation** — simulate what-if shock propagation through the graph
- **Explainability** — attention extraction, GNNExplainer attribution, causal tracing
- **Calibrated uncertainty** — conformal prediction intervals out of the box

---

## Install

```bash
pip install git+https://github.com/Gaurang111/graphoracle.git

# Optional extras
pip install "graphoracle[viz] @ git+https://github.com/Gaurang111/graphoracle.git"       # Plotly + matplotlib visualisations
pip install "graphoracle[benchmarks] @ git+https://github.com/Gaurang111/graphoracle.git" # Benchmark dataset loaders
pip install "graphoracle[all] @ git+https://github.com/Gaurang111/graphoracle.git"        # Everything
```

> **Requirements:** Python 3.10+, PyTorch 2.1+, PyTorch Geometric 2.4+

---

## Quickstart (60 seconds)

The fastest way to try GraphOracle is with the built-in synthetic data generator — no CSV files needed.

```python
from graphoracle import GraphOracle
from graphoracle.domains import ElectricGridDomain
from graphoracle.connectors import SyntheticGenerator

# 1. Pick a domain (handles schema, node/edge types for you)
domain = ElectricGridDomain()

# 2. Generate a synthetic graph to explore with immediately
graph = SyntheticGenerator(domain.schema).generate(
    nodes_per_type={"substation": 10, "weather_station": 5,
                    "renewable_source": 5, "industrial_consumer": 8,
                    "residential_zone": 8},
    num_timesteps=48,   # 48 hours of hourly readings
)

# 3. Train
oracle = GraphOracle(model="graphoracle", horizons=[1, 6, 24])
oracle.fit(graph, epochs=50)

# 4. Forecast every node across all horizons
forecast = oracle.predict(graph)
print(forecast.summary())
```

---

## Using Your Own Data (CSV)

Once you have real data, load it directly from CSV files:

```python
from graphoracle import GraphOracle
from graphoracle.domains import ElectricGridDomain

domain = ElectricGridDomain()

graph = domain.build_graph_from_csv(
    nodes="data/nodes.csv",
    edges="data/edges.csv",
    timeseries="data/readings.csv",
)

# Inject an external event (e.g. an incoming storm)
graph.inject_event(
    event_type="weather_event",
    affected_nodes=["substation_12", "substation_15"],
    features={"severity": 0.8, "wind_speed_ms": 28.0},
    start="2024-08-01 06:00",
    end="2024-08-01 18:00",
)

oracle = GraphOracle(model="graphoracle", horizons=[1, 6, 24])
oracle.fit(graph, epochs=50)

forecast = oracle.predict(graph)
print(forecast.summary())
```

---

## Custom Model (Level 1 — Swap the Backbone)

Drop in any PyTorch module as the GNN backbone. GraphOracle handles everything else (data prep, training loop, evaluation, checkpointing, uncertainty).

```python
from graphoracle import GraphOracle
import torch.nn as nn

class MyGATBackbone(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 128), nn.ReLU(),
            nn.Linear(128, out_channels),
        )

    def forward(self, x, edge_index):
        # x: (N, in_channels) — GraphOracle projects raw features for you
        return self.fc(x)

oracle = GraphOracle(
    backbone=MyGATBackbone,
    backbone_kwargs={"in_channels": 64, "out_channels": 64},
    horizons=[1, 6, 24],
)
oracle.fit(graph, epochs=50)
```

> GraphOracle automatically projects each node type's raw feature dimension to `in_channels` before calling your backbone — you never have to match schema dimensions manually.

---

## Custom Model (Level 2 — Full Control)

Subclass `BaseForecastModel` for complete control over the forward pass. GraphOracle still handles training, evaluation, and checkpointing.

```python
from graphoracle.models import BaseForecastModel, ModelRegistry
import torch.nn as nn

class MyFullModel(BaseForecastModel):
    def __init__(self, schema, horizons, **kwargs):
        super().__init__(schema, horizons, **kwargs)
        self.encoders = nn.ModuleDict({
            nt.name: nn.Linear(schema.node_dim(nt.name), 64)
            for nt in schema.node_types
        })
        self.heads = nn.ModuleDict({
            nt.name: nn.ModuleDict({
                f"h{h}": nn.Linear(64, schema.target_dim(nt.name))
                for h in horizons
            })
            for nt in schema.forecast_node_types
        })

    def forward(self, graph, node_features, edge_index, temporal_encoding, memory=None):
        # node_features: {node_type: Tensor(N, T, F)}
        out = {}
        for nt in graph.schema.forecast_node_types:
            feat = node_features[nt.name].mean(1)   # (N, T, F) → (N, F)
            h = self.encoders[nt.name](feat).relu()
            out[nt.name] = {
                horizon: self.heads[nt.name][f"h{horizon}"](h)
                for horizon in self.horizons
            }
        return out

    def required_history_steps(self):
        return 24

# Register globally — use by name anywhere in your codebase
ModelRegistry.register("my_model", MyFullModel)

oracle = GraphOracle(model="my_model", horizons=[1, 6, 24])
oracle.fit(graph, epochs=50)
```

---

## Custom Domain (~30 lines)

Define a completely new domain by describing its node types, features, and edges.

```python
from graphoracle.domains import BaseDomain
from graphoracle.graph import GraphSchema, NodeType, EdgeType

class WaterSystemDomain(BaseDomain):
    @property
    def schema(self) -> GraphSchema:
        return GraphSchema(
            node_types=[
                NodeType("reservoir",    features=["volume_ml", "inflow_rate"],  targets=["volume_ml"]),
                NodeType("pump_station", features=["flow_ls", "energy_kwh"],     targets=["flow_ls"]),
                NodeType("demand_zone",  features=["population", "demand_ls"],   targets=["demand_ls"]),
            ],
            edge_types=[
                EdgeType("pipe", "reservoir",    "pump_station"),
                EdgeType("pipe", "pump_station", "demand_zone"),
            ],
        )

domain = WaterSystemDomain()
graph = domain.build_synthetic_graph(num_timesteps=48)
```

---

## Built-in Models

| Model | Architecture | Best for |
|-------|-------------|----------|
| `graphoracle` | HGT + TGN memory + adaptive graph + conformal PI | Production use, full power |
| `hgt` | Heterogeneous Graph Transformer (3 layers) | Heterogeneous graphs, fast training |
| `tgn` | Temporal Graph Network | Dynamic graphs, event streams |
| `stgnn` | Spatiotemporal GNN (DCRNN-style) | Spatial sensor networks |
| `lstm` | Per-type LSTM | Simple baseline |
| `gru` | Per-type GRU | Simple baseline |

---

## Built-in Domains

| Domain | Node Types | Use Case |
|--------|-----------|----------|
| `ElectricGridDomain` | Substation, Weather, Renewable, Consumer, Market | Grid load/generation forecasting |
| `SupplyChainDomain` | Supplier, Manufacturer, Warehouse, Retailer, Port | Supply chain risk & inventory |
| `TrafficWeatherDomain` | Traffic sensor, Weather station, Road segment | Urban traffic speed/flow |
| `PandemicDomain` | Region, Hospital, Mobility hub, Variant | Epidemic spread forecasting |
| `FinancialDomain` | Bank, Asset class, Sector, Macro indicator | Financial contagion & credit risk |

---

## Training API

```python
from graphoracle import GraphOracle
from graphoracle.training import TrainingConfig

config = TrainingConfig(
    epochs=100,
    learning_rate=1e-3,
    loss="quantile",             # "mae" | "rmse" | "mape" | "quantile"
    scheduler="cosine",
    early_stopping_patience=10,
    use_curriculum=True,
    checkpoint_dir="checkpoints/",
    device="cuda",               # or "cpu"
)

oracle = GraphOracle(model="graphoracle", horizons=[1, 6, 24])
history = oracle.fit(graph, config=config)
history.plot()
```

---

## Cascade Simulation

```python
# What if substation_7 loses 40% capacity?
cascade = oracle.simulate_cascade(
    graph,
    shocks=[
        {"node": "substation_7", "feature": "load_mw", "change": -0.40, "type": "percent"}
    ],
    steps=24,
)
print(cascade.highest_risk_nodes(k=10))
print(cascade.estimated_recovery_hours())
```

---

## Explainability

```python
explanation = oracle.explain(graph, node_id="substation_12", horizon=24)

# Which neighbours influenced this forecast most?
print(explanation.top_influencers(k=5))

# Feature attribution scores
print(explanation.feature_importances())

# Trace what caused an anomaly
chains = explanation.causal_trace(anomaly_timestamp="2024-08-01 14:00")
```

---

## Architecture

```
Input: HeterogeneousTemporalGraph (N nodes, T timesteps, F features per type)
                        │
          ┌─────────────▼─────────────┐
          │  1. NodeEncoder            │  type-specific Linear + temporal PE
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │  2. TGN Memory Module      │  GRU-updated per-node memory
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │  3. Heterogeneous Graph    │  4-layer HGT
          │     Transformer (HGT)      │  type-specific W_Q, W_K, W_V
          └─────────────┬─────────────┘
                        │
                        │  Note: the 4-layer default is tunable via `num_layers`.
                        │  For dense graphs, 2–3 layers are recommended to reduce
                        │  oversmoothing (see Limitations).
                        │
          ┌─────────────▼─────────────┐
          │  4. Adaptive Graph         │  α·A_schema + (1-α)·A_learned
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │  5. Global Context         │  graph-level pooling → cross-attention
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │  6. Multi-Horizon Heads    │  per type · per horizon · quantile
          └───────────────────────────┘
Output: {node_type: {horizon: predictions}}
```

---

## Limitations

- **Fixed graph structure.** Nodes and edges must be defined upfront. If the graph topology changes — new nodes added, edges removed — the model requires retraining from scratch or fine-tuning on the updated graph.

- **Manual domain definition required.** There is no automatic type inference from raw CSV data. Every new domain requires writing a schema class that defines node types, edge types, and feature lists. The 5 built-in domains cover common cases; anything else requires this manual step.

- **Feature encoding is the user's responsibility.** The framework expects float vectors as input. String fields, booleans, categoricals, and nested objects must be encoded by the user before data enters the framework. There is no built-in preprocessing pipeline for mixed data types.

- **Oversmoothing risk at deep layers.** The default 4-layer HGT means each node aggregates information from up to 4 hops away. In dense graphs this causes node embeddings to converge toward similar values, which degrades forecast quality. Using 2–3 layers is recommended for dense graphs.

- **Synchronous message passing does not capture real propagation delays.** All nodes update simultaneously each round. Real-world cascading effects — power failures, epidemic spread — propagate with actual time delays that this architecture cannot represent directly.

- **Cascade simulation is approximated.** `simulate_cascade` propagates shocks through learned GNN embeddings, not through a physics-based or mechanistic model. Results indicate which nodes are likely affected but do not explain causal pathways.

- **Conformal prediction intervals quantify uncertainty range but do not explain its source.** A wide interval at a specific node could indicate genuine volatility or insufficient training data for that node type. The intervals are not decomposed by cause.

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Citation

If you use GraphOracle in your research, please cite:

```bibtex
@software{graphoracle2024,
  title   = {{GraphOracle}: Multi-domain Heterogeneous Temporal Graph Forecasting Framework},
  author  = {GraphOracle Contributors},
  year    = {2026},
  url     = {https://github.com/Gaurang111/graphoracle},
  license = {MIT},
}
```

---

## License

MIT — see [LICENSE](LICENSE).
