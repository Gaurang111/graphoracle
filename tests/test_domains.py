"""Tests for all domain definitions."""

from __future__ import annotations

import pytest

from graphoracle.graph.schema import GraphSchema


class _DomainTestBase:
    domain_cls = None

    def _domain(self):
        return self.domain_cls()

    def test_schema_is_graph_schema(self):
        d = self._domain()
        assert isinstance(d.schema, GraphSchema)

    def test_has_node_types(self):
        d = self._domain()
        assert len(d.schema.node_types) > 0

    def test_has_edge_types(self):
        d = self._domain()
        assert len(d.schema.edge_types) > 0

    def test_default_horizons_nonempty(self):
        d = self._domain()
        assert len(d.default_horizons) > 0

    def test_has_at_least_one_forecast_node_type(self):
        d = self._domain()
        assert len(d.schema.forecast_node_types) > 0

    def test_edge_types_reference_valid_node_types(self):
        d = self._domain()
        schema = d.schema
        node_names = set(schema.node_type_names)
        for et in schema.edge_types:
            assert et.src_type in node_names, f"{et.src_type} not in schema"
            assert et.dst_type in node_names, f"{et.dst_type} not in schema"


class TestTrafficWeatherDomain(_DomainTestBase):
    from graphoracle.domains.traffic_weather import TrafficWeatherDomain
    domain_cls = TrafficWeatherDomain

    def test_has_traffic_sensor(self):
        d = self._domain()
        names = d.schema.node_type_names
        assert "traffic_sensor" in names

    def test_traffic_sensor_has_targets(self):
        d = self._domain()
        nt = d.schema.get_node_type("traffic_sensor")
        assert len(nt.targets) > 0


class TestElectricGridDomain(_DomainTestBase):
    from graphoracle.domains.electric_grid import ElectricGridDomain
    domain_cls = ElectricGridDomain

    def test_has_substation(self):
        d = self._domain()
        assert "substation" in d.schema.node_type_names

    def test_substation_has_targets(self):
        d = self._domain()
        nt = d.schema.get_node_type("substation")
        assert len(nt.targets) > 0


class TestSupplyChainDomain(_DomainTestBase):
    from graphoracle.domains.supply_chain import SupplyChainDomain
    domain_cls = SupplyChainDomain


class TestPandemicDomain(_DomainTestBase):
    from graphoracle.domains.pandemic import PandemicDomain
    domain_cls = PandemicDomain


class TestFinancialDomain(_DomainTestBase):
    from graphoracle.domains.financial import FinancialDomain
    domain_cls = FinancialDomain
