"""Generate procurement recommendations using EOQ/ROP logic."""

from __future__ import annotations

import logging
import math
import os
from statistics import mean
from typing import Any, Dict, List

import numpy as np

from ..core.config import load_yaml
from ..models.schemas import ForecastResponse, ReorderRec
from .inventory_service import InventoryService

LOGGER = logging.getLogger(__name__)


def _z_value_for_service_level(service_level: float) -> float:
    """Return the z-score for a one-sided service level using erfcinv."""

    service_level = max(0.5, min(service_level, 0.999))
    try:
        from math import erfcinv, sqrt

        return sqrt(2.0) * erfcinv(2.0 * (1.0 - service_level))
    except Exception:  # pragma: no cover - erfcinv unavailable in some interpreters
        # Fallback mapping for common levels (rounded)
        lookup = {
            0.9: 1.2816,
            0.95: 1.6449,
            0.97: 1.8808,
            0.98: 2.0537,
            0.99: 2.3263,
        }
        closest = min(lookup.keys(), key=lambda x: abs(x - service_level))
        return lookup[closest]


class ProcurementService:
    """EOQ/ROP based recommendation engine."""

    def __init__(
        self,
        config_root: str = "configs",
        inventory_service: InventoryService | None = None,
    ) -> None:
        thresholds_path = os.path.join(config_root, "thresholds.yaml")
        settings_path = os.path.join(config_root, "settings.yaml")

        thresholds = load_yaml(thresholds_path)
        settings = load_yaml(settings_path)

        self.auto_approval_limit = float(thresholds.get("auto_approval_limit", 1_000.0))
        self.min_service_level = float(thresholds.get("min_service_level", 0.9))
        self.gmroi_min = float(thresholds.get("gmroi_min", 1.0))
        self.max_cash_outlay = float(thresholds.get("max_cash_outlay", 10_000.0))

        self.carrying_cost_rate = float(settings.get("carrying_cost_rate", 0.20))
        self.service_level_target = float(settings.get("service_level_target", settings.get("default_service_level", 0.95)))
        self.default_lead_time_days = float(settings.get("lead_time_days", 7))
        self.default_order_cost = float(settings.get("order_cost", 75.0))
        self.default_margin_rate = float(settings.get("gross_margin_rate", 0.30))

        self.inventory_service = inventory_service or InventoryService()

    # ------------------------------------------------------------------
    def _derive_daily_stats(self, forecast: ForecastResponse) -> tuple[float, float, float]:
        means = np.array([max(pt.mean, 0.0) for pt in forecast.forecast], dtype=float)
        confidences = [pt.confidence for pt in forecast.forecast]
        interval_spreads = np.array(
            [max(pt.hi - pt.lo, 0.0) for pt in forecast.forecast],
            dtype=float,
        )

        if means.size == 0:
            return 0.0, 0.0, 0.0

        daily_mean = float(np.mean(means))
        # Convert interval width (approx 80% interval) to std: width ≈ 2*z*std
        # using the same z as forecasting service (1.28155) -> std ≈ width / (2*z)
        z_interval = 1.28155
        derived_stds = interval_spreads / max(2.0 * z_interval, 1e-6)
        daily_std = float(np.mean(derived_stds)) if np.any(interval_spreads) else float(np.std(means))
        avg_conf = float(mean(confidences)) if confidences else 0.0
        return daily_mean, max(daily_std, 0.0), avg_conf

    # ------------------------------------------------------------------
    def recommend(self, forecast: ForecastResponse, context: Dict[str, Any]) -> List[ReorderRec]:
        if not forecast.forecast:
            LOGGER.warning("Forecast for SKU %s is empty; no recommendation generated", forecast.sku_id)
            return []

        daily_mean, daily_std, avg_conf = self._derive_daily_stats(forecast)
        lead_time_days = float(context.get("lead_time_days", self.default_lead_time_days))
        if lead_time_days <= 0:
            lead_time_days = self.default_lead_time_days

        service_level = float(context.get("service_level_target", self.service_level_target))
        z_val = _z_value_for_service_level(service_level)
        safety_stock = z_val * daily_std * math.sqrt(lead_time_days)
        reorder_point = int(round(daily_mean * lead_time_days + safety_stock))

        unit_cost = float(context.get("unit_cost", self.inventory_service.estimate_unit_cost(forecast.sku_id)))
        order_cost = float(context.get("order_cost", self.default_order_cost))
        carrying_cost = unit_cost * self.carrying_cost_rate

        annual_demand = daily_mean * 365.0
        if annual_demand <= 0 or carrying_cost <= 0:
            eoq = 0.0
        else:
            eoq = math.sqrt((2.0 * annual_demand * order_cost) / carrying_cost)

        order_qty = int(max(round(eoq), 0))
        if order_qty == 0 and daily_mean > 0:
            order_qty = max(int(round(daily_mean * lead_time_days)), 1)

        total_spend = order_qty * unit_cost
        gross_margin_rate = float(context.get("gross_margin_rate", self.default_margin_rate))
        expected_gmroi = (gross_margin_rate * unit_cost) / max(carrying_cost, 1e-6)
        gmroi_delta = expected_gmroi - self.gmroi_min

        requires_approval = (
            total_spend > self.auto_approval_limit
            or service_level < self.min_service_level
            or gmroi_delta < 0
            or total_spend > self.max_cash_outlay
        )

        # Confidence penalises variability and approval overrides
        variability_penalty = 1.0 / (1.0 + (daily_std / (daily_mean + 1e-6))) if daily_mean else 0.3
        confidence = float(max(0.1, min(0.99, avg_conf * variability_penalty)))
        if requires_approval:
            confidence = min(confidence, 0.6)

        LOGGER.info(
            "Procurement rec for %s: mean=%.2f std=%.2f lead=%.1f order_qty=%s spend=%.2f gmroi_delta=%.2f",
            forecast.sku_id,
            daily_mean,
            daily_std,
            lead_time_days,
            order_qty,
            total_spend,
            gmroi_delta,
        )

        return [
            ReorderRec(
                sku_id=forecast.sku_id,
                reorder_point=max(reorder_point, 0),
                order_qty=max(order_qty, 0),
                gmroi_delta=round(gmroi_delta, 4),
                confidence=round(confidence, 4),
                requires_approval=requires_approval,
            )
        ]