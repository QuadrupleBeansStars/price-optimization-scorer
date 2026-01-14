"""
Enhanced evaluation script that returns detailed breakdown for Streamlit app.
"""

import math
import pandas as pd
from typing import Dict, Tuple


def price_elasticity(cat: str) -> float:
    return {
        "Beverages": -1.3,
        "Snacks": -1.1,
        "Dairy": -0.9,
        "PersonalCare": -0.6,
    }.get(cat, -1.0)


def allowed_ending(p: float) -> float:
    ip = math.floor(p)
    cand = [ip + c for c in [0.00, 0.50, 0.90]] + [ip + 1 + c for c in [0.00, 0.50, 0.90]]
    return min(cand, key=lambda x: abs(x - p))


def load_data(data_dir: str):
    """Load all required data files from the dataset directory."""
    sales = pd.read_csv(f"{data_dir}/sales_history.csv")
    price = pd.read_csv(f"{data_dir}/price_cost.csv")
    inv = pd.read_csv(f"{data_dir}/inventory.csv")
    cal = pd.read_csv(f"{data_dir}/calendar_weather.csv")
    sku = pd.read_csv(f"{data_dir}/sku_master.csv")
    store = pd.read_csv(f"{data_dir}/store_master.csv")
    comp = pd.read_csv(f"{data_dir}/competitor_prices.csv")
    xel = pd.read_csv(f"{data_dir}/XEL.csv")

    return sales, price, inv, cal, sku, store, comp, xel


def simulate_with_breakdown(
    proposed: pd.DataFrame,
    data_dir: str,
) -> Dict:
    """
    Run simulation and return detailed breakdown for Streamlit display.

    Returns:
        dict with keys:
            - final_score: float
            - total_profit: float
            - stockout_penalty: float
            - violations: int
            - instability: int
            - violation_details: list of dicts (optional)
            - daily_metrics: DataFrame (optional)
    """
    sales, price, inv, cal, sku, store, comp, xel = load_data(data_dir)

    horizon_dates = sorted(proposed["date"].unique())

    sku_meta = sku.merge(price, on="sku_id")
    sku_cat = sku_meta.set_index("sku_id")["category"].to_dict()
    sku_reg = sku_meta.set_index("sku_id")["regular_price"].to_dict()
    unit_cost = sku_meta.set_index("sku_id")["unit_cost"].to_dict()
    vat = sku_meta.set_index("sku_id")["vat_rate"].to_dict()

    # cross-elasticity lookup
    xel_mat = {
        (int(r.sku_i), int(r.sku_j)): float(r.xel_ij)
        for _, r in xel.iterrows()
    }

    # base qty estimation
    sales_m = sales.merge(price, on="sku_id", how="left")
    sales_m["near_reg"] = (
        abs(sales_m["price_paid"] - sales_m["regular_price"]) <= 0.2
    )

    base = (
        sales_m[sales_m["near_reg"]]
        .groupby(["store_id", "sku_id"])["qty"]
        .mean()
        .rename("base_qty")
        .reset_index()
    )

    base_qty = {
        (int(r.store_id), int(r.sku_id)): float(r.base_qty)
        for _, r in base.iterrows()
    }

    def fallback(cat):
        return {
            "Beverages": 6.0,
            "Snacks": 4.0,
            "Dairy": 3.0,
            "PersonalCare": 1.6,
        }.get(cat, 3.0)

    inv_state = inv.set_index(["store_id", "sku_id"])["on_hand"].to_dict()
    comp_map = {
        (int(r.sku_id), r.date): float(r.comp_price)
        for _, r in comp.iterrows()
    }

    # Load lead times for replenishment
    lead_times = inv.set_index(["store_id", "sku_id"])["lead_time_days"].to_dict()

    # Track pending replenishments
    pending_orders = {}
    REORDER_QTY_DAYS = 7

    total_profit = 0.0
    stockouts = 0
    violations = 0
    instability = 0.0
    last_price = {}

    # Detailed tracking
    violation_details = []
    daily_metrics = []

    for ds in horizon_dates:
        # Process replenishments arriving today
        for (s, sk, delivery_date), qty in list(pending_orders.items()):
            if delivery_date == ds:
                key = (s, sk)
                inv_state[key] = inv_state.get(key, 0) + qty
                del pending_orders[(s, sk, delivery_date)]

        row = cal[cal["date"] == ds]

        if row.empty:
            dow, is_pay, rain, is_holiday, temp = (0, 0, 0.2, 0, 30.0)
        else:
            rr = row.iloc[0]
            dow = int(rr.dow)
            is_pay = int(rr.is_payday)
            rain = float(rr.rain_index)
            is_holiday = int(rr.is_holiday)
            temp = float(rr.temp)

        todays = proposed[proposed["date"] == ds]

        day_profit = 0.0
        day_stockouts = 0.0

        for _, r in todays.iterrows():
            store_id = int(r.store_id)
            sku_id = int(r.sku_id)
            price_p = float(r.proposed_price)

            reg = sku_reg.get(sku_id, price_p)
            cost = unit_cost.get(sku_id, 0.0)
            vatr = vat.get(sku_id, 0.07)

            # Track violations
            if price_p < cost * (1 + vatr) - 1e-6:
                violations += 1
                violation_details.append({
                    'date': ds,
                    'store_id': store_id,
                    'sku_id': sku_id,
                    'type': 'Below Cost',
                    'price': price_p,
                    'min_allowed': cost * (1 + vatr),
                })

            if (reg - price_p) / max(reg, 0.01) > 0.30 + 1e-6:
                violations += 1
                violation_details.append({
                    'date': ds,
                    'store_id': store_id,
                    'sku_id': sku_id,
                    'type': 'Over 30% Discount',
                    'price': price_p,
                    'regular_price': reg,
                    'discount': (reg - price_p) / reg,
                })

            if abs(allowed_ending(price_p) - price_p) > 0.009:
                violations += 1
                violation_details.append({
                    'date': ds,
                    'store_id': store_id,
                    'sku_id': sku_id,
                    'type': 'Invalid Ending',
                    'price': price_p,
                    'should_be': allowed_ending(price_p),
                })

            key = (store_id, sku_id)

            if key in last_price and abs(last_price[key] - price_p) > 1e-6:
                instability += 1

            last_price[key] = price_p

            # demand model
            baseq = base_qty.get(
                key, fallback(sku_cat.get(sku_id, "Beverages"))
            )
            eps = price_elasticity(sku_cat.get(sku_id, "Beverages"))

            comp_p = comp_map.get((sku_id, ds), reg)
            comp_gap = (price_p - comp_p) / max(comp_p, 0.01)

            q = baseq * (price_p / max(reg, 0.1)) ** eps
            q *= (1 + 0.08 * (1 - comp_gap))
            q *= (1 + 0.05 * is_pay)
            q *= (1 + 0.03 * (dow in (5, 6)))
            q *= (1 + 0.10 * is_holiday)
            q *= (1 - 0.06 * rain)

            # Temperature effects
            cat = sku_cat.get(sku_id, "Beverages")
            if cat == "Beverages":
                q *= (1 + 0.015 * (temp - 30))
            elif cat == "Dairy":
                q *= (1 - 0.01 * (temp - 30))

            # cross elasticity
            rels = [j for (i, j) in xel_mat.keys() if i == sku_id][:5]
            for j in rels:
                pj_row = todays[
                    (todays["store_id"] == store_id)
                    & (todays["sku_id"] == j)
                ]
                pj = (
                    float(pj_row["proposed_price"].values[0])
                    if not pj_row.empty
                    else sku_reg.get(j, price_p)
                )

                x = xel_mat[(sku_id, j)]
                q *= (
                    1
                    + x
                    * (
                        (reg - price_p) / max(reg, 0.01)
                        - (sku_reg.get(j, reg) - pj)
                        / max(sku_reg.get(j, reg), 0.01)
                    )
                )

            # Replenishment logic
            on_hand = inv_state.get(key, 0)
            avg_daily_demand = base_qty.get(
                key, fallback(sku_cat.get(sku_id, "Beverages"))
            )
            lead = lead_times.get(key, 3)

            inventory_days = on_hand / max(avg_daily_demand, 0.1)
            reorder_threshold_days = max(3, lead + 1)

            if inventory_days < reorder_threshold_days:
                reorder_qty = int(avg_daily_demand * REORDER_QTY_DAYS)
                current_idx = horizon_dates.index(ds)
                if current_idx + lead < len(horizon_dates):
                    delivery_date = horizon_dates[current_idx + lead]
                    if not any(
                        (s, sk, d) == (store_id, sku_id, delivery_date)
                        for (s, sk, d) in pending_orders.keys()
                    ):
                        pending_orders[(store_id, sku_id, delivery_date)] = reorder_qty

            sell = min(on_hand, max(0, int(round(q))))

            profit_this = (price_p - cost) * sell
            stockout_this = price_p * (round(q) - sell) if sell < int(round(q)) else 0

            total_profit += profit_this
            stockouts += stockout_this
            day_profit += profit_this
            day_stockouts += stockout_this

            inv_state[key] = on_hand - sell

        # Track daily metrics
        daily_metrics.append({
            'date': ds,
            'profit': day_profit,
            'stockout_penalty': day_stockouts,
        })

    score = (
        total_profit
        - stockouts
        - ((violations / 1000.0) * abs(total_profit))
        - 5.0 * instability
    )

    return {
        'final_score': score,
        'total_profit': total_profit,
        'stockout_penalty': stockouts,
        'violations': violations,
        'instability': int(instability),
        'violation_details': violation_details,
        'daily_metrics': pd.DataFrame(daily_metrics),
    }
