# Writing Custom Domains

A domain bundles a `GraphSchema` with optional data-loading helpers.
You can define a complete new domain in ~30 lines.

---

## Minimal Example

```python
from graphoracle.domains import BaseDomain
from graphoracle.graph import GraphSchema, NodeType, EdgeType

class WaterSystemDomain(BaseDomain):

    @property
    def schema(self) -> GraphSchema:
        return GraphSchema(
            node_types=[
                NodeType(
                    name="reservoir",
                    features=["volume_ml", "inflow_rate", "outflow_rate", "elevation_m"],
                    targets=["volume_ml", "outflow_rate"],
                ),
                NodeType(
                    name="pump_station",
                    features=["capacity_ls", "current_flow", "energy_kwh", "pressure_bar"],
                    targets=["current_flow", "energy_kwh"],
                ),
                NodeType(
                    name="demand_zone",
                    features=["population", "demand_ls", "time_of_day", "season"],
                    targets=["demand_ls"],
                ),
                NodeType(
                    name="rainfall_sensor",
                    features=["mm_per_hour", "temperature", "humidity"],
                    targets=[],   # context node, not a forecast target
                ),
            ],
            edge_types=[
                EdgeType("pipe_flow", "reservoir", "pump_station"),
                EdgeType("pipe_flow_out", "pump_station", "demand_zone"),
                EdgeType("weather_influence", "rainfall_sensor", "reservoir"),
            ],
        )

    @property
    def default_horizons(self):
        return [1, 6, 24]   # hourly steps

# Use it
domain = WaterSystemDomain()
graph = domain.build_synthetic_graph(
    nodes_per_type={"reservoir": 3, "pump_station": 5,
                    "demand_zone": 8, "rainfall_sensor": 2},
    num_timesteps=168,  # 1 week of hourly data
)
```

---

## Loading from CSVs

The default `build_graph_from_csv()` expects:

**nodes.csv**
```
node_id,node_type,feature1,feature2,...
```

**edges.csv**
```
src_id,dst_id,edge_type
```

**timeseries.csv** (long format)
```
node_id,timestamp,feature1,feature2,...
```

Override `build_graph_from_csv` to customise parsing:

```python
class WaterSystemDomain(BaseDomain):
    @property
    def schema(self): ...

    def build_graph_from_csv(self, nodes, edges, timeseries=None, **kwargs):
        # Custom parsing logic
        import pandas as pd
        reservoir_df = pd.read_csv(nodes["reservoir"])
        # ... your domain-specific loading ...
        return graph
```

---

## Schema Design Tips

- **Features** = all inputs the model sees (include context, metadata, historical signals)
- **Targets** = what you want to forecast (subset of features)
- Set `targets=[]` for purely contextual nodes (e.g. weather stations)
- Use `spatial_features=["lat", "lon"]` to enable positional encoding
- Edge features (optional) can encode capacity, weight, direction, etc.

---

## Domain Checklist

- [ ] Schema has at least one node type with non-empty `targets`
- [ ] All edge `src_type` and `dst_type` exist in `node_types`
- [ ] Feature names are unique within a node type
- [ ] `default_horizons` matches the temporal resolution of your data
