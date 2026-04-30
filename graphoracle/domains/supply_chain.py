"""Supply chain domain definition."""

from __future__ import annotations

from graphoracle.domains.base import BaseDomain
from graphoracle.graph.schema import EdgeType, GraphSchema, NodeType


class SupplyChainDomain(BaseDomain):
    """
    Heterogeneous graph schema for supply chain risk and inventory forecasting.

    Node types: supplier, manufacturer, warehouse, retailer, port
    Edge types: material_flow, shipment, storage_link, distribution
    """

    @property
    def schema(self) -> GraphSchema:
        return GraphSchema(
            node_types=[
                NodeType(
                    "supplier",
                    features=["capacity_units", "lead_time_days", "reliability_score", "cost_per_unit"],
                    targets=["capacity_units", "lead_time_days"],
                ),
                NodeType(
                    "manufacturer",
                    features=["throughput_units", "wip_inventory", "utilisation_rate", "defect_rate"],
                    targets=["throughput_units", "wip_inventory"],
                ),
                NodeType(
                    "warehouse",
                    features=["stock_units", "capacity_units", "holding_cost", "stockout_rate"],
                    targets=["stock_units"],
                ),
                NodeType(
                    "retailer",
                    features=["demand_units", "sales_units", "backorder_rate", "price"],
                    targets=["demand_units", "sales_units"],
                ),
                NodeType(
                    "port",
                    features=["throughput_teu", "congestion_index", "delay_days", "capacity_teu"],
                    targets=["throughput_teu", "delay_days"],
                ),
            ],
            edge_types=[
                EdgeType("material_flow", "supplier", "manufacturer"),
                EdgeType("shipment", "manufacturer", "warehouse"),
                EdgeType("distribution", "warehouse", "retailer"),
                EdgeType("port_export", "manufacturer", "port"),
                EdgeType("port_import", "port", "warehouse"),
                EdgeType("direct_supply", "supplier", "warehouse"),
            ],
        )

    @property
    def default_horizons(self) -> list[int]:
        return [1, 7, 30]
