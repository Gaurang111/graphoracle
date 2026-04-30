"""
Bring Your Own Domain
=====================
Shows how to define a completely custom domain in ~30 lines.

Run: python examples/bring_your_own_domain.py
"""

from graphoracle import GraphOracle
from graphoracle.domains import BaseDomain
from graphoracle.graph import EdgeType, GraphSchema, NodeType
from graphoracle.training import TrainingConfig


# ──────────────────────────────────────────────────────────────────────────────
# 1. Define the domain (~30 lines)
# ──────────────────────────────────────────────────────────────────────────────

class DataCenterDomain(BaseDomain):
    """
    Domain for data centre power and thermal forecasting.

    Nodes: server rack, cooling unit, UPS, PDU (power distribution unit)
    Edges: power supply, cooling connection, load feeds
    """

    @property
    def schema(self) -> GraphSchema:
        return GraphSchema(
            node_types=[
                NodeType(
                    name="server_rack",
                    features=["cpu_util", "mem_util", "power_kw", "temp_c", "pue"],
                    targets=["power_kw", "temp_c"],
                ),
                NodeType(
                    name="cooling_unit",
                    features=["inlet_temp", "outlet_temp", "flow_rate", "power_kw"],
                    targets=["power_kw", "outlet_temp"],
                ),
                NodeType(
                    name="ups",
                    features=["load_kva", "battery_pct", "efficiency"],
                    targets=["load_kva"],
                ),
                NodeType(
                    name="pdu",
                    features=["total_load_kw", "phase_a_kw", "phase_b_kw", "phase_c_kw"],
                    targets=["total_load_kw"],
                ),
            ],
            edge_types=[
                EdgeType("powers", "pdu", "server_rack"),
                EdgeType("powers", "ups", "pdu"),      # NOTE: same edge name is fine
                EdgeType("cools", "cooling_unit", "server_rack"),
                EdgeType("rack_to_cooling", "server_rack", "cooling_unit"),
            ],
        )

    @property
    def default_horizons(self):
        return [1, 6, 24]   # hourly steps


# ──────────────────────────────────────────────────────────────────────────────
# 2. Use it exactly like a built-in domain
# ──────────────────────────────────────────────────────────────────────────────

domain = DataCenterDomain()
print("Domain schema:")
print(domain.schema)
print()

graph = domain.build_synthetic_graph(
    nodes_per_type={
        "server_rack": 20,
        "cooling_unit": 4,
        "ups": 2,
        "pdu": 4,
    },
    num_timesteps=48,
    seed=2024,
)
print(graph.summary())

# 3. Inject a cooling failure event
graph.inject_event(
    event_type="cooling_failure",
    affected_nodes=["cooling_unit_0"],
    features={"outlet_temp": 5.0, "flow_rate": -0.8},
    start="2024-01-01 14:00",
    end="2024-01-01 20:00",
)

# 4. Train and predict
oracle = GraphOracle(
    model="graphoracle",
    horizons=domain.default_horizons,
)
history = oracle.fit(graph, config=TrainingConfig(epochs=5))
print(f"\nTraining: best val loss = {history.best_val_loss():.4f}")

forecast = oracle.predict(graph)
df = forecast.to_dataframe()
print(f"\nForecast DataFrame shape: {df.shape}")
print("Sample rows:")
print(df[df["node_type"] == "server_rack"].head(6).to_string(index=False))

# 5. Simulate a UPS overload cascade
cascade = oracle.simulate_cascade(
    graph,
    shocks=[{"node": "ups_0", "feature": "load_kva", "change": 0.9, "type": "percent"}],
    steps=6,
)
print("\nCascade — top 5 most affected nodes:")
for n in cascade.highest_risk_nodes(k=5):
    print(f"  {n}")
