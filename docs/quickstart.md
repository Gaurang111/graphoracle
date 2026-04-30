# Quickstart

## Installation

```bash
git clone https://github.com/Gaurang111/graphoracle
cd graphoracle
pip install -e .

# Optional extras
pip install -e ".[viz]"          # Plotly + matplotlib visualisations
pip install -e ".[benchmarks]"   # Benchmark dataset loaders
pip install -e ".[all]"          # Everything
```

> **Requirements:** Python 3.10+, PyTorch 2.1+, PyTorch Geometric 2.4+

---

## 1. Build a Graph

### Synthetic data (fastest way to get started)

```python
from graphoracle.domains import ElectricGridDomain
from graphoracle.connectors import SyntheticGenerator

domain = ElectricGridDomain()
graph = SyntheticGenerator(domain.schema).generate(
    nodes_per_type={"substation": 5, "weather_station": 3,
                    "renewable_source": 3, "industrial_consumer": 4,
                    "residential_zone": 4},
    num_timesteps=48,
)
```

### From CSV files

```python
graph = domain.build_graph_from_csv(
    nodes="nodes.csv",
    edges="edges.csv",
    timeseries="readings.csv",
)
```

### From scratch with the fluent builder

```python
from graphoracle.graph import GraphBuilder, GraphSchema, NodeType, EdgeType
import numpy as np

schema = GraphSchema(
    node_types=[NodeType("sensor", features=["value"], targets=["value"])],
    edge_types=[EdgeType("link", "sensor", "sensor")],
)

graph = (
    GraphBuilder(schema)
    .add_nodes("sensor", ["s0", "s1", "s2"], np.random.randn(3, 24, 1))
    .add_edges("link", ["s0", "s1"], ["s1", "s2"])
    .build()
)
```

---

## 2. Inject Events

```python
graph.inject_event(
    event_type="storm",
    affected_nodes=["s0", "s1"],
    features={"severity": 0.9},
    start="2024-08-01 06:00",
    end="2024-08-01 18:00",
)
```

---

## 3. Train

```python
from graphoracle import GraphOracle
from graphoracle.training import TrainingConfig

oracle = GraphOracle(model="graphoracle", horizons=[1, 6, 24])
config = TrainingConfig(epochs=50, loss="quantile", device="cpu")
history = oracle.fit(graph, config=config)
history.plot()
```

---

## 4. Predict

```python
forecast = oracle.predict(graph)
print(forecast.summary())
df = forecast.to_dataframe()
```

---

## 5. Evaluate

```python
results = oracle.evaluate(graph, metrics=["MAE", "RMSE", "MAPE"])
print(results.summary_table())
```

---

## 6. Explain

```python
explanation = oracle.explain(graph, node_id="substation_0", horizon=6)
print(explanation.top_influencers(k=3))
print(explanation.feature_importances())
```

---

## 7. Simulate a Cascade

```python
cascade = oracle.simulate_cascade(
    graph,
    shocks=[{"node": "substation_0", "feature": "load_mw", "change": -0.5, "type": "percent"}],
    steps=12,
)
print(cascade.highest_risk_nodes(k=5))
```
