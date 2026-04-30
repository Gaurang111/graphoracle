"""Pandemic / epidemic spread domain definition."""

from __future__ import annotations

from graphoracle.domains.base import BaseDomain
from graphoracle.graph.schema import EdgeType, GraphSchema, NodeType


class PandemicDomain(BaseDomain):
    """
    Heterogeneous graph schema for epidemic spread forecasting.

    Node types: region, hospital, mobility_hub, variant
    Edge types: transmission, patient_flow, mobility_link, variant_spread
    """

    @property
    def schema(self) -> GraphSchema:
        return GraphSchema(
            node_types=[
                NodeType(
                    "region",
                    features=[
                        "cases_daily", "deaths_daily", "recovered_daily",
                        "vaccination_rate", "population_density", "r_effective",
                    ],
                    targets=["cases_daily", "deaths_daily"],
                ),
                NodeType(
                    "hospital",
                    features=[
                        "icu_occupancy", "bed_utilisation", "admissions_daily",
                        "staff_available", "ventilators_in_use",
                    ],
                    targets=["icu_occupancy", "admissions_daily"],
                ),
                NodeType(
                    "mobility_hub",
                    features=["passenger_volume", "connectivity_index", "screening_rate"],
                    targets=["passenger_volume"],
                ),
                NodeType(
                    "variant",
                    features=["transmissibility", "severity_index", "immune_escape", "prevalence"],
                    targets=["prevalence"],
                ),
            ],
            edge_types=[
                EdgeType("transmission", "region", "region"),
                EdgeType("patient_flow", "region", "hospital"),
                EdgeType("mobility_link", "mobility_hub", "region"),
                EdgeType("variant_spread", "variant", "region"),
                EdgeType("hub_variant", "mobility_hub", "variant"),
            ],
        )

    @property
    def default_horizons(self) -> list[int]:
        return [1, 7, 14]
