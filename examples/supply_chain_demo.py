"""
Supply Chain Disruption Demo
============================
Demonstrates cascade simulation on a supply chain graph.

Run: python examples/supply_chain_demo.py
"""

from graphoracle import GraphOracle
from graphoracle.domains import SupplyChainDomain
from graphoracle.training import TrainingConfig

# Build domain and synthetic graph
domain = SupplyChainDomain()
graph = domain.build_synthetic_graph(
    nodes_per_type={
        "supplier": 5,
        "manufacturer": 3,
        "warehouse": 4,
        "retailer": 8,
        "port": 2,
        "transport_link": 3,
        "external_risk": 2,
    },
    num_timesteps=30,
    seed=99,
)
print(graph.summary())

# Train
oracle = GraphOracle(model="lstm", horizons=[1, 7, 14])
oracle.fit(graph, config=TrainingConfig(epochs=5))

# Forecast
forecast = oracle.predict(graph)
print(forecast.summary())

# Simulate a supplier disruption
print("\nSimulating: supplier_0 output drops 60%...")
cascade = oracle.simulate_cascade(
    graph,
    shocks=[
        {"node": "supplier_0", "feature": "current_output",
         "change": -0.60, "type": "percent"},
    ],
    steps=10,
)

print("\nHighest risk nodes (top 5):")
for node in cascade.highest_risk_nodes(k=5):
    print(f"  {node}")

recovery = cascade.estimated_recovery_hours(threshold=0.05)
if recovery:
    print(f"\nEstimated recovery steps for first 5 nodes:")
    for node, steps in list(recovery.items())[:5]:
        print(f"  {node}: {steps} steps")
