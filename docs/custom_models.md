# Writing Custom Models

GraphOracle provides three levels of customisation — from swapping a single
backbone layer to writing a complete architecture.

---

## Level 1 — Swap the Backbone (simplest)

Provide any `nn.Module` that accepts `(x, edge_index)` and returns `(N, out_channels)`.
GraphOracle wraps it in a complete forecasting pipeline automatically.

```python
from graphoracle import GraphOracle
import torch.nn as nn

class MyBackbone(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 128), nn.ReLU(),
            nn.Linear(128, out_channels),
        )

    def forward(self, x, edge_index):
        return self.fc(x)

oracle = GraphOracle(
    backbone=MyBackbone,
    backbone_kwargs={"in_channels": 64, "out_channels": 64},
    horizons=[1, 6, 24],
)
oracle.fit(graph, epochs=50)
```

**What graphoracle handles automatically:**
- Temporal pooling of the input window
- Separate forecast heads per horizon and node type
- Training loop, loss, checkpointing, evaluation

---

## Level 2 — Subclass `BaseForecastModel` (full control)

```python
from graphoracle.models import BaseForecastModel
from graphoracle.graph import HeterogeneousTemporalGraph
import torch
import torch.nn as nn

class MyFullModel(BaseForecastModel):
    """
    Full control over encoding, message passing, and decoding.
    graphoracle handles: data loading, training loop, evaluation,
    checkpointing, uncertainty wrapping, and explainability hooks.
    User handles: the actual forward pass.
    """

    def __init__(self, schema, horizons, hidden=64, **kwargs):
        super().__init__(schema, horizons, **kwargs)
        # schema tells you node types, feature dims, edge types
        self.encoders = nn.ModuleDict({
            nt.name: nn.Linear(schema.node_dim(nt.name), hidden)
            for nt in schema.node_types
        })
        self.heads = nn.ModuleDict({
            nt.name: nn.ModuleDict({
                f"h{h}": nn.Linear(hidden, schema.target_dim(nt.name))
                for h in horizons
            })
            for nt in schema.forecast_node_types
        })

    def forward(
        self,
        graph: HeterogeneousTemporalGraph,
        node_features: dict,
        edge_index: dict,
        temporal_encoding: torch.Tensor,
        memory: dict | None = None,
    ) -> dict:
        """
        Returns: {node_type: {horizon: predictions}}
        graphoracle expects this exact output format.
        """
        out = {}
        for nt in graph.schema.forecast_node_types:
            feat = node_features[nt.name]
            if feat.ndim == 3:
                feat = feat.mean(1)            # temporal pool (N, T, F) → (N, F)
            h = torch.relu(self.encoders[nt.name](feat))
            out[nt.name] = {
                horizon: self.heads[nt.name][f"h{horizon}"](h)
                for horizon in self.horizons
            }
        return out

    def required_history_steps(self) -> int:
        """Tell graphoracle how many past steps your model needs."""
        return 24

    def supports_missing_nodes(self) -> bool:
        return False
```

### Lifecycle Hooks

Override these optional methods for setup work:

```python
def on_fit_start(self, graph):
    """Called once before training begins — initialise stateful components here."""
    self.memory = torch.zeros(graph.num_nodes("sensor"), 64)

def on_predict_start(self, graph):
    """Called before each inference run."""
    pass

def reset_memory(self):
    """Reset any persistent state (called by oracle between runs)."""
    self.memory.zero_()
```

---

## Level 3 — Register Globally and Share

```python
from graphoracle.models import ModelRegistry

# Decorator syntax
@ModelRegistry.register("transformer_v2")
class TransformerV2(BaseForecastModel):
    def __init__(self, schema, horizons, num_heads=8, dropout=0.1, **kw):
        super().__init__(schema, horizons)
        # ...

    def forward(self, ...): ...
    def required_history_steps(self): return 48

# Use anywhere by string name
oracle = GraphOracle(model="transformer_v2", horizons=[1, 6, 24])

# Or from a config dict (great for experiments / sweeps)
oracle = GraphOracle.from_config({
    "model": "transformer_v2",
    "horizons": [1, 6, 24],
    "model_kwargs": {"num_heads": 8, "dropout": 0.1},
})
```

---

## The `forward()` Contract

Your `forward()` must return:

```python
{
    "node_type_name": {
        1:  torch.Tensor,   # shape (N, target_dim) or (N, target_dim * n_quantiles)
        6:  torch.Tensor,
        24: torch.Tensor,
    },
    ...
}
```

Only include node types that have forecast targets in the schema.
Node types with empty `targets=[]` should be omitted from the output.

---

## Tips

- Use `self.schema.node_types` and `self.schema.edge_types` to discover the graph structure at build time
- `schema.node_dim(nt_name)` gives feature dimensionality
- `schema.target_dim(nt_name)` gives target dimensionality
- For quantile outputs, return `(N, target_dim * n_quantiles)` tensors — graphoracle will extract percentiles automatically
- If your model is memory-intensive, override `required_history_steps()` to return a smaller window
