"""Generate procurement recommendations using EOQ/ROP logic."""

from __future__ import annotations

import logging
import math
import os
from numbers import Real
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from ..core.config import load_yaml
from ..models.schemas import ForecastResponse, ReorderRec
from .inventory_service import InventoryService

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def z_for_service_level(service_level: float) -> float:
    """Return the z-score associated with a one-sided service level."""

    level = float(service_level)
    if not math.isfinite(level):
        level = 0.95
    level = max(0.5, min(level, 0.999))

    try:  # Prefer numerical inverse if available
        from math import erfcinv, sqrt

        return sqrt(2.0) * erfcinv(2.0 * (1.0 - level))
    except Exception:  # pragma: no cover - fallback for limited interpreters
        lookup = {
            0.90: 1.2816,
            0.95: 1.6449,
            0.97: 1.8808,
            0.98: 2.0537,
            0.99: 2.3263,
        }
        closest = min(lookup.keys(), key=lambda x: abs(x - level))
        return lookup[closest]


def calculate_daily_stats(points: Iterable[Tuple[float, float]]) -> Tuple[float, float]:
    """Return (mean, std) demand estimates from (mean, confidence) tuples."""

    means = np.array([max(mu, 0.0) for mu, _ in points], dtype=float)
    if means.size == 0:
        return 0.0, 0.0

    daily_mean = float(np.mean(means))
    # ``np.std`` defaults to population std (ddof=0) which we use here
    daily_std = float(np.std(means)) if means.size > 1 else 0.0
    return max(daily_mean, 0.0), max(daily_std, 0.0)


def calculate_eoq(
    daily_mean: float,
    unit_cost: float,
    order_cost: float,
    carrying_cost_rate: float,
) -> float:
    """Compute the economic order quantity using annual demand."""

    if daily_mean <= 0 or unit_cost <= 0 or carrying_cost_rate <= 0 or order_cost <= 0:
        return 0.0

    annual_demand = daily_mean * 365.0
    carrying_cost = carrying_cost_rate * unit_cost
    if carrying_cost <= 0:
        return 0.0

    value = (2.0 * order_cost * annual_demand) / carrying_cost
    return math.sqrt(value) if value > 0 else 0.0


def calculate_rop(
    daily_mean: float,
    daily_std: float,
    lead_time_days: float,
    service_level: float,
) -> Tuple[float, float, float]:
    """Return reorder point, lead-time mean and std."""

    lead_time = max(lead_time_days, 0.0)
    mu_l = max(daily_mean, 0.0) * lead_time
    sigma_l = max(daily_std, 0.0) * math.sqrt(lead_time)
    z_value = z_for_service_level(service_level)
    reorder_point = mu_l + z_value * sigma_l
    return reorder_point, mu_l, sigma_l


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

        self.carrying_cost_rate = float(settings.get("carrying_cost_rate", 0.20))
        self.service_level_target = float(
            settings.get("service_level_target", settings.get("default_service_level", 0.95))
        )
        self.default_lead_time_days = float(settings.get("lead_time_days", 7.0))
        self.default_order_cost = float(
            settings.get("order_setup_cost", settings.get("order_cost", 50.0))
        )
        self.default_margin_rate = float(settings.get("gross_margin_rate", 0.30))

        self.inventory_service = inventory_service or InventoryService()

    # ------------------------------------------------------------------
    def _forecast_statistics(self, forecast: ForecastResponse) -> Tuple[float, float, float]:
        """Derive daily demand stats and overall confidence from the forecast."""

        means_conf = [(pt.mean, pt.confidence) for pt in forecast.forecast]
        daily_mean, daily_std = calculate_daily_stats(means_conf)
        confidences = [max(min(pt.confidence, 1.0), 0.0) for pt in forecast.forecast]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return daily_mean, daily_std, avg_conf

    # ------------------------------------------------------------------
    def recommend(self, forecast: ForecastResponse, context: Dict[str, Any]) -> List[ReorderRec]:
        if not forecast.forecast:
            LOGGER.warning(
                "Forecast for SKU %s is empty; no recommendation generated", forecast.sku_id
            )
            return []

        daily_mean, daily_std, forecast_confidence = self._forecast_statistics(forecast)
        if daily_mean <= 1e-6 and daily_std <= 1e-6:
            LOGGER.info(
                "Forecast for SKU %s has negligible demand; skipping recommendation",
                forecast.sku_id,
            )
            return []

        sku_id = forecast.sku_id
        service_level = float(context.get("service_level_target", self.service_level_target))

        lead_time = context.get("lead_time_days")
        if lead_time is None:
            get_lead_time = getattr(self.inventory_service, "get_lead_time_days", None)
            lead_time = get_lead_time(sku_id) if callable(get_lead_time) else None
        lead_time = float(lead_time) if lead_time is not None else self.default_lead_time_days
        if lead_time <= 0:
            lead_time = self.default_lead_time_days

        unit_cost = context.get("unit_cost")
        if unit_cost is None:
            get_unit_cost = getattr(self.inventory_service, "get_unit_cost", None)
            if callable(get_unit_cost):
                unit_cost = get_unit_cost(sku_id)
            else:
                estimate_cost = getattr(self.inventory_service, "estimate_unit_cost", None)
                unit_cost = estimate_cost(sku_id) if callable(estimate_cost) else None
        unit_cost = float(unit_cost) if unit_cost is not None else 1.0
        if unit_cost <= 0:
            unit_cost = 1.0

        order_cost = float(context.get("order_cost", self.default_order_cost))

        reorder_point_val, mu_l, sigma_l = calculate_rop(
            daily_mean=daily_mean,
            daily_std=daily_std,
            lead_time_days=lead_time,
            service_level=service_level,
        )

        eoq_val = calculate_eoq(
            daily_mean=daily_mean,
            unit_cost=unit_cost,
            order_cost=order_cost,
            carrying_cost_rate=self.carrying_cost_rate,
        )

        order_qty = math.ceil(eoq_val) if eoq_val > 0 else 0
        safety_floor = 0
        if mu_l > 0:
            buffer_multiplier = 0.5 if sigma_l > 0 else 0.0
            safety_floor = math.ceil(mu_l * buffer_multiplier)
        if safety_floor > order_qty:
            order_qty = safety_floor
        if order_qty <= 0 and daily_mean > 0:
            order_qty = max(math.ceil(mu_l), 1)

        get_inventory = getattr(self.inventory_service, "get_current_inventory", None)
        current_inventory = get_inventory(sku_id) if callable(get_inventory) else 0
        if isinstance(current_inventory, Real):
            current_inventory = float(current_inventory)
        else:
            try:
                current_inventory = float(current_inventory)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Inventory service returned non-numeric inventory for %s; defaulting to 0",
                    sku_id,
                )
                current_inventory = 0.0
        if not math.isfinite(current_inventory) or current_inventory < 0:
            current_inventory = 0.0
        inventory_units_proxy = max(current_inventory, 1.0)
        get_latest_price = getattr(self.inventory_service, "get_latest_price", None)
        latest_price = get_latest_price(sku_id) if callable(get_latest_price) else None
        sell_price = None if latest_price is None else float(latest_price.sell_price)
        if not sell_price or sell_price <= 0:
            sell_price = unit_cost * (1.0 + self.default_margin_rate)

        margin_ratio = (sell_price - unit_cost) / unit_cost if unit_cost else 0.0
        gmroi_delta = margin_ratio * (order_qty / inventory_units_proxy)

        total_spend = order_qty * unit_cost
        cash_budget = context.get("cash_budget")
        over_budget = False
        if cash_budget is not None:
            try:
                over_budget = float(total_spend) > float(cash_budget)
            except Exception:
                over_budget = False

        requires_approval = (
            total_spend > self.auto_approval_limit
            or over_budget
            or gmroi_delta < self.gmroi_min
            or service_level < self.min_service_level
        )

        confidence = max(min(forecast_confidence, 0.99), 0.01)
        if requires_approval:
            confidence = min(confidence, 0.6)
        reorder_point = int(max(round(reorder_point_val), 0))
        order_qty_int = int(max(order_qty, 0))
        gmroi_delta_val = float(gmroi_delta if math.isfinite(gmroi_delta) else 0.0)

        LOGGER.info(
            "Procurement rec for %s: mean=%.2f std=%.2f lead=%.1f order_qty=%d spend=%.2f gmroi=%.3f",
            sku_id,
            daily_mean,
            daily_std,
            lead_time,
            order_qty_int,
            total_spend,
            gmroi_delta_val,
        )

        return [
            ReorderRec(
                sku_id=sku_id,
                reorder_point=reorder_point,
                order_qty=order_qty_int,
                gmroi_delta=round(gmroi_delta_val, 4),
                confidence=round(confidence, 4),
                requires_approval=requires_approval,
            )
        ]
