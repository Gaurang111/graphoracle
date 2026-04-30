"""Electric grid domain definition."""

from __future__ import annotations

from graphoracle.domains.base import BaseDomain
from graphoracle.graph.schema import EdgeType, GraphSchema, NodeType


class ElectricGridDomain(BaseDomain):
    """
    Heterogeneous graph schema for electric grid forecasting.

    Node types: substation, weather_station, renewable_source,
                industrial_consumer, residential_zone,
                transmission_line, market_signal, event_node
    Edge types: power_flow, weather_influence, market_dependency,
                grid_coupling, renewable_feed
    """

    @property
    def schema(self) -> GraphSchema:
        return GraphSchema(
            node_types=[
                NodeType(
                    "substation",
                    features=["load_mw", "voltage_kv", "frequency_hz", "reactive_power_mvar"],
                    targets=["load_mw", "voltage_kv"],
                ),
                NodeType(
                    "weather_station",
                    features=["temperature_c", "wind_speed_ms", "solar_irradiance", "humidity"],
                    targets=[],
                ),
                NodeType(
                    "renewable_source",
                    features=["output_mw", "capacity_mw", "availability", "ramp_rate"],
                    targets=["output_mw"],
                ),
                NodeType(
                    "industrial_consumer",
                    features=["demand_mw", "peak_demand", "power_factor", "shift_flexibility"],
                    targets=["demand_mw"],
                ),
                NodeType(
                    "residential_zone",
                    features=["demand_mw", "population", "time_of_day_factor", "season_factor"],
                    targets=["demand_mw"],
                ),
                NodeType(
                    "transmission_line",
                    features=["flow_mw", "capacity_mw", "congestion_rate"],
                    targets=["flow_mw"],
                ),
                NodeType(
                    "market_signal",
                    features=["price_per_mwh", "demand_forecast", "reserve_margin"],
                    targets=[],
                ),
                NodeType(
                    "event_node",
                    features=["severity", "duration_h", "affected_area"],
                    targets=[],
                ),
            ],
            edge_types=[
                EdgeType("power_flow", "substation", "substation"),
                EdgeType("weather_influence", "weather_station", "substation"),
                EdgeType("weather_influence_renewable", "weather_station", "renewable_source"),
                EdgeType("renewable_feed", "renewable_source", "substation"),
                EdgeType("consumer_load", "industrial_consumer", "substation"),
                EdgeType("residential_load", "residential_zone", "substation"),
                EdgeType("line_flow", "substation", "transmission_line"),
                EdgeType("market_dependency", "market_signal", "substation"),
                EdgeType("event_impact", "event_node", "substation"),
            ],
        )

    @property
    def default_horizons(self) -> list[int]:
        return [1, 6, 24]
