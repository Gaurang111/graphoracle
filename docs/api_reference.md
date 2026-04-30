# API Reference

## Core

### `GraphOracle`

```python
GraphOracle(
    model: str = "graphoracle",
    horizons: list[int] = [1, 6, 24],
    backbone: type | None = None,
    backbone_kwargs: dict | None = None,
    device: str = "cpu",
    **model_kwargs,
)
```

| Method | Description |
|--------|-------------|
| `fit(graph, epochs, config, val_graph)` | Train the model |
| `predict(graph, reference_time, horizon_unit_hours)` | Run inference → `ForecastResult` |
| `evaluate(graph, metrics, ...)` | Compute MAE/RMSE/MAPE/CRPS → `EvalResult` |
| `explain(graph, node_id, horizon)` | Return `_ExplainProxy` |
| `simulate_cascade(graph, shocks, steps)` | Shock propagation → `CascadeResult` |
| `save(path)` | Save model weights |
| `load(path, schema)` | Load weights into a new oracle |
| `from_config(config_dict)` | Factory from dict |

---

## Graph

### `GraphSchema`
```python
GraphSchema(node_types: list[NodeType], edge_types: list[EdgeType])
```
Pydantic model. Validates edge type references on construction.

### `NodeType`
```python
NodeType(name, features, targets, spatial_features=[], metadata={})
```

### `EdgeType`
```python
EdgeType(name, src_type, dst_type, features=[], metadata={})
```

### `HeterogeneousTemporalGraph`
```python
graph.add_nodes(node_type, node_ids, features, timestamps)
graph.add_edges(edge_type, src_ids, dst_ids, edge_features)
graph.inject_event(event_type, affected_nodes, features, start, end)
graph.get_node_features(node_type)   # → (N, T, F) Tensor
graph.get_edge_index(edge_type)      # → (2, E) Tensor
graph.to(device)
graph.clone()
graph.summary()
```

### `GraphBuilder`
Fluent builder: `.add_nodes().add_edges().inject_event().build()`

---

## Models

### `BaseForecastModel`
ABC. Subclass and implement `forward()` and `required_history_steps()`.

### `ModelRegistry`
```python
ModelRegistry.register("name", ModelClass)   # or @ModelRegistry.register("name")
ModelRegistry.get("name")    # → class
ModelRegistry.available()    # → list[str]
```

Built-in models: `graphoracle`, `hgt`, `lstm`, `gru`

---

## Training

### `TrainingConfig`
```python
TrainingConfig(
    epochs=100, learning_rate=1e-3, loss="mae",
    scheduler="cosine", early_stopping_patience=15,
    use_curriculum=False, checkpoint_dir="checkpoints/",
    device="cpu", ...
)
```
Loss options: `"mae"`, `"rmse"`, `"mape"`, `"quantile"`, `"pinball"`

### `TrainingHistory`
```python
history.train_losses   # list[float]
history.val_losses     # list[float]
history.best_val_loss()
history.plot()
```

---

## Forecasting

### `ForecastResult`
```python
result.summary()
result.get(node_id, feature)    # → NodeForecast
result.to_dataframe()           # → pd.DataFrame
result.plot_gantt(node, feature)
result.all_nodes()
```

### `CascadeResult`
```python
result.highest_risk_nodes(k)
result.estimated_recovery_hours(threshold)
result.impact_delta(step)
result.plot_impact_over_time()
```

---

## Domains

All domains inherit `BaseDomain`:
```python
domain.schema                # GraphSchema
domain.default_horizons      # list[int]
domain.build_graph_from_csv(nodes, edges, timeseries)
domain.build_synthetic_graph(nodes_per_type, num_timesteps, seed)
```

---

## Connectors

```python
CSVConnector(schema).load(nodes, edges, timeseries)
JSONConnector(schema).load(path)
DataFrameConnector(schema).load(node_dfs, edge_dfs, ...)
SyntheticGenerator(schema, seed).generate(nodes_per_type, num_timesteps)
```

---

## Knowledge

```python
EdgeDiscovery(method="correlation", threshold=0.5).discover(node_ids, timeseries)
# methods: "correlation", "granger", "spatial", "mutual_info"

EventInjector().apply(graph)
EventInjector().build_event_tensor(graph, node_type, feature_names)
```

---

## Explainability

```python
GNNExplainer(model).explain_node(graph, node_id, node_type, horizon)
# → NodeImportance(feature_scores, neighbour_scores)

AttentionExtractor(model).register_hooks().get_attention_weights()

CausalTracer(max_hops=4).trace(graph, anomaly_node_id, node_type)
# → list[CausalChain]
```

---

## Benchmarks

```python
MetrLA.load(num_timesteps)
PemsBay.load(num_timesteps)
NREL118.load(num_timesteps)
ETTDataset.load(split, csv_path, num_timesteps)

Evaluator(oracle, metrics=["MAE", "RMSE", "MAPE", "CRPS"]).run(graph)
# → EvalResult
```
