"""
Electric Grid Quickstart
========================
Demonstrates the full pipeline on a synthetic electric grid graph.
Run with: python examples/electric_grid_quickstart.py
"""

from graphoracle import GraphOracle
from graphoracle.domains import ElectricGridDomain
from graphoracle.training import TrainingConfig

# 1. Build domain and synthetic graph
domain = ElectricGridDomain()
graph = domain.build_synthetic_graph(
    nodes_per_type={
        "substation": 10,
        "weather_station": 3,
        "renewable_source": 4,
        "industrial_consumer": 5,
        "residential_zone": 5,
        "transmission_line": 3,
        "market_signal": 2,
        "event_node": 1,
    },
    num_timesteps=48,
    seed=42,
)
print(graph.summary())

# 2. Inject a weather event
graph.inject_event(
    event_type="storm",
    affected_nodes=["substation_0", "substation_1"],
    features={"load_mw": -0.3},
    start="2024-01-01 06:00",
    end="2024-01-01 18:00",
)

# 3. Train
oracle = GraphOracle(model="graphoracle", horizons=[1, 6, 24])
config = TrainingConfig(
    epochs=5,    # increase for real use
    loss="mae",
    device="cpu",
    checkpoint_dir="checkpoints/grid_demo",
)
history = oracle.fit(graph, config=config)
print(f"\nTraining complete — best val loss: {history.best_val_loss():.4f}")

# 4. Forecast
forecast = oracle.predict(graph)
print("\n" + forecast.summary())

df = forecast.to_dataframe()
print("\nFirst 10 forecast rows:")
print(df.head(10).to_string(index=False))

# 5. Evaluate
results = oracle.evaluate(graph, metrics=["MAE", "RMSE"])
print("\n" + results.summary_table())

# 6. Explain a node
explanation = oracle.explain(graph, node_id="substation_0", horizon=6)
print("\nTop influencers for substation_0 @ horizon 6:")
for node, score in explanation.top_influencers(k=3):
    print(f"  {node}: {score:.3f}")

# 7. Cascade simulation
print("\nSimulating cascade from a substation_0 capacity drop...")
cascade = oracle.simulate_cascade(
    graph,
    shocks=[{"node": "substation_0", "feature": "load_mw",
              "change": -0.4, "type": "percent"}],
    steps=6,
)
print("Highest risk nodes:", cascade.highest_risk_nodes(k=5))
print("Estimated recovery (steps):", cascade.estimated_recovery_hours(threshold=0.1))
