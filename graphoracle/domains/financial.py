"""Financial contagion and credit risk domain definition."""

from __future__ import annotations

from graphoracle.domains.base import BaseDomain
from graphoracle.graph.schema import EdgeType, GraphSchema, NodeType


class FinancialDomain(BaseDomain):
    """
    Heterogeneous graph schema for financial contagion and credit-risk forecasting.

    Node types: bank, asset_class, sector, macro_indicator
    Edge types: interbank_exposure, portfolio_overlap, sector_dependency,
                macro_influence
    """

    @property
    def schema(self) -> GraphSchema:
        return GraphSchema(
            node_types=[
                NodeType(
                    "bank",
                    features=[
                        "capital_ratio", "leverage_ratio", "npls_pct",
                        "liquidity_coverage", "net_interest_margin",
                    ],
                    targets=["capital_ratio", "npls_pct"],
                ),
                NodeType(
                    "asset_class",
                    features=["return_pct", "volatility", "sharpe_ratio", "drawdown_max"],
                    targets=["return_pct", "volatility"],
                ),
                NodeType(
                    "sector",
                    features=["gdp_contribution", "employment_idx", "credit_growth", "default_rate"],
                    targets=["default_rate", "credit_growth"],
                ),
                NodeType(
                    "macro_indicator",
                    features=["value", "yoy_change", "forecast_deviation"],
                    targets=[],
                ),
            ],
            edge_types=[
                EdgeType("interbank_exposure", "bank", "bank"),
                EdgeType("portfolio_overlap", "bank", "asset_class"),
                EdgeType("sector_lending", "bank", "sector"),
                EdgeType("sector_asset", "sector", "asset_class"),
                EdgeType("macro_bank", "macro_indicator", "bank"),
                EdgeType("macro_sector", "macro_indicator", "sector"),
            ],
        )

    @property
    def default_horizons(self) -> list[int]:
        return [1, 5, 20]
