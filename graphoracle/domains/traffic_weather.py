"""Traffic and weather sensor domain definition."""

from __future__ import annotations

from graphoracle.domains.base import BaseDomain
from graphoracle.graph.schema import EdgeType, GraphSchema, NodeType


class TrafficWeatherDomain(BaseDomain):
    """
    Heterogeneous graph schema for urban traffic speed/flow forecasting.

    Node types: traffic_sensor, weather_station, road_segment
    Edge types: road_link, weather_road_influence, sensor_coverage
    """

    @property
    def schema(self) -> GraphSchema:
        return GraphSchema(
            node_types=[
                NodeType(
                    "traffic_sensor",
                    features=["speed_mph", "flow_veh_h", "occupancy", "density", "incident_flag"],
                    targets=["speed_mph", "flow_veh_h"],
                ),
                NodeType(
                    "weather_station",
                    features=["temperature_c", "precipitation_mm", "visibility_m", "wind_speed_ms"],
                    targets=[],
                ),
                NodeType(
                    "road_segment",
                    features=["length_km", "lanes", "speed_limit_mph", "surface_condition"],
                    targets=["surface_condition"],
                ),
            ],
            edge_types=[
                EdgeType("road_link", "traffic_sensor", "traffic_sensor"),
                EdgeType("weather_road", "weather_station", "traffic_sensor"),
                EdgeType("sensor_coverage", "traffic_sensor", "road_segment"),
            ],
        )

    @property
    def default_horizons(self) -> list[int]:
        return [1, 3, 12]
