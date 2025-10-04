"""
Service responsible for generating procurement recommendations.

This module computes reorder quantities and points based on forecast data and
business guardrails.  For demonstration purposes it implements a very
simple calculation: reorder enough to cover a week of forecast demand
plus a safety margin, and multiply by a factor to determine the order
quantity.  The recommendation includes a confidence score and whether
human approval is required based on configured spending thresholds.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any
import numpy as np

from ..models.schemas import ForecastResponse, ReorderRec
from ..core.config import load_yaml


class ProcurementService:
    def __init__(self, config_root: str = "configs") -> None:
        """Load threshold configuration from YAML files."""
        thresholds_path = os.path.join(config_root, "thresholds.yaml")
        thresholds = load_yaml(thresholds_path)
        # Set defaults if keys are missing
        self.auto_approval_limit = float(thresholds.get("auto_approval_limit", 1000))
        self.min_service_level = float(thresholds.get("min_service_level", 0.9))
        self.min_gmroi_delta = float(thresholds.get("min_gmroi_delta", 0.05))
        self.max_cash_outlay = float(thresholds.get("max_cash_outlay", 10000))

    def recommend(self, forecast: ForecastResponse, context: Dict[str, Any]) -> List[ReorderRec]:
        """Generate a reorder recommendation from forecast data.

        Args:
            forecast: The demand forecast for a SKU.
            context: Additional business context (unused in this demo).

        Returns:
            A list containing a single ``ReorderRec``.
        """
        # Calculate expected demand over the horizon
        demand_values = [point.mean for point in forecast.forecast]
        if not demand_values:
            reorder_point = 0
        else:
            # Use one week of demand as reorder point
            reorder_point = int(np.mean(demand_values[:7]) * 7)
        # Order enough to cover two reorder points
        order_qty = max(reorder_point * 2, 1)
        # Compute a simple GMROI delta: encourage orders only if demand is positive
        gmroi_delta = 0.1 if reorder_point > 0 else 0.0
        # Confidence is based on the average forecast confidence
        confidences = [point.confidence for point in forecast.forecast]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        # Determine whether human approval is required: if the order's notional cost exceeds the auto approval limit
        # Since we do not know unit price, treat quantity as a proxy for cost
        requires_approval = order_qty > self.auto_approval_limit
        return [
            ReorderRec(
                sku_id=forecast.sku_id,
                reorder_point=reorder_point,
                order_qty=order_qty,
                gmroi_delta=gmroi_delta,
                confidence=avg_conf,
                requires_approval=requires_approval,
            )
        ]