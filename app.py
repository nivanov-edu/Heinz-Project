import copy
import io
import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats


st.set_page_config(
    page_title="Heinz Logistics Dashboard",
    page_icon="📦",
    layout="wide",
)


WEEKDAY_ORDER = [
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
    "Monday",
]
WEEKDAY_SHORT_MAP = {
    "Tuesday": "Tu",
    "Wednesday": "We",
    "Thursday": "Th",
    "Friday": "Fr",
    "Saturday": "Sa",
    "Sunday": "Su",
    "Monday": "Mo",
}
SHORT_TO_FULL = {v: k for k, v in WEEKDAY_SHORT_MAP.items()}
MODE_COLORS = {
    "mixed": "#2E74B5",
    "intermodal": "#70AD47",
    "truck": "#ED7D31",
}
EXPECTED_COLUMNS = {"Weekday", "Brutto Weight", "Cbm"}


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

NOTEBOOK_SUMMARY_ROWS = [
    {
        "Rule": "Rule 01: Every x days (mixed)",
        "Optimal x": 7,
        "x description": "every 7 days",
        "Avg Total Cost (€)": 20396.84,
        "Avg Storage Cost (€)": 2568.04,
        "Avg Transport Cost (€)": 17828.80,
        "Avg Shipments": 13.0,
        "Avg CO₂ Total (kg)": 6758.70,
        "Avg CO₂ Rail (kg)": 3006.30,
        "Avg CO₂ Road (kg)": 3752.40,
        "Avg CO₂ per Shipment (kg)": 519.90,
        "Avg CO₂ per kg Cargo (g)": 37.47,
    },
    {
        "Rule": "Rule 02: Fixed weekday(s) (mixed)",
        "Optimal x": "Mo",
        "x description": "Mo",
        "Avg Total Cost (€)": 20396.84,
        "Avg Storage Cost (€)": 2568.04,
        "Avg Transport Cost (€)": 17828.80,
        "Avg Shipments": 13.0,
        "Avg CO₂ Total (kg)": 6758.70,
        "Avg CO₂ Rail (kg)": 3006.30,
        "Avg CO₂ Road (kg)": 3752.40,
        "Avg CO₂ per Shipment (kg)": 519.90,
        "Avg CO₂ per kg Cargo (g)": 37.47,
    },
    {
        "Rule": "Rule 03: kg threshold (mixed)",
        "Optimal x": 7178.00,
        "x description": "7178.0 kg",
        "Avg Total Cost (€)": 24644.99,
        "Avg Storage Cost (€)": 1327.09,
        "Avg Transport Cost (€)": 23317.90,
        "Avg Shipments": 18.5,
        "Avg CO₂ Total (kg)": 5492.90,
        "Avg CO₂ Rail (kg)": 3815.60,
        "Avg CO₂ Road (kg)": 1677.30,
        "Avg CO₂ per Shipment (kg)": 296.90,
        "Avg CO₂ per kg Cargo (g)": 30.95,
    },
    {
        "Rule": "Rule 04: cbm threshold (mixed)",
        "Optimal x": 34.69,
        "x description": "34.7 cbm",
        "Avg Total Cost (€)": 25985.96,
        "Avg Storage Cost (€)": 1090.56,
        "Avg Transport Cost (€)": 24895.40,
        "Avg Shipments": 19.4,
        "Avg CO₂ Total (kg)": 5668.80,
        "Avg CO₂ Rail (kg)": 3811.80,
        "Avg CO₂ Road (kg)": 1857.00,
        "Avg CO₂ per Shipment (kg)": 292.00,
        "Avg CO₂ per kg Cargo (g)": 31.82,
    },
    {
        "Rule": "Rule 05: Every x days (intermodal)",
        "Optimal x": 7,
        "x description": "every 7 days",
        "Avg Total Cost (€)": 29649.04,
        "Avg Storage Cost (€)": 2568.04,
        "Avg Transport Cost (€)": 27081.00,
        "Avg Shipments": 13.0,
        "Avg CO₂ Total (kg)": 6048.60,
        "Avg CO₂ Rail (kg)": 4241.30,
        "Avg CO₂ Road (kg)": 1807.40,
        "Avg CO₂ per Shipment (kg)": 465.30,
        "Avg CO₂ per kg Cargo (g)": 34.08,
    },
    {
        "Rule": "Rule 06: Fixed weekday(s) (intermodal)",
        "Optimal x": "Mo",
        "x description": "Mo",
        "Avg Total Cost (€)": 29649.04,
        "Avg Storage Cost (€)": 2568.04,
        "Avg Transport Cost (€)": 27081.00,
        "Avg Shipments": 13.0,
        "Avg CO₂ Total (kg)": 6048.60,
        "Avg CO₂ Rail (kg)": 4241.30,
        "Avg CO₂ Road (kg)": 1807.40,
        "Avg CO₂ per Shipment (kg)": 465.30,
        "Avg CO₂ per kg Cargo (g)": 34.08,
    },
    {
        "Rule": "Rule 07: kg threshold (intermodal)",
        "Optimal x": 10110.83,
        "x description": "10110.8 kg",
        "Avg Total Cost (€)": 28587.56,
        "Avg Storage Cost (€)": 2530.56,
        "Avg Transport Cost (€)": 26057.00,
        "Avg Shipments": 14.6,
        "Avg CO₂ Total (kg)": 6014.00,
        "Avg CO₂ Rail (kg)": 4217.00,
        "Avg CO₂ Road (kg)": 1797.00,
        "Avg CO₂ per Shipment (kg)": 411.90,
        "Avg CO₂ per kg Cargo (g)": 33.87,
    },
    {
        "Rule": "Rule 08: cbm threshold (intermodal)",
        "Optimal x": 58.59,
        "x description": "58.6 cbm",
        "Avg Total Cost (€)": 27511.92,
        "Avg Storage Cost (€)": 3199.92,
        "Avg Transport Cost (€)": 24312.00,
        "Avg Shipments": 12.6,
        "Avg CO₂ Total (kg)": 5904.60,
        "Avg CO₂ Rail (kg)": 4140.20,
        "Avg CO₂ Road (kg)": 1764.30,
        "Avg CO₂ per Shipment (kg)": 469.30,
        "Avg CO₂ per kg Cargo (g)": 33.24,
    },
    {
        "Rule": "Rule 09: Every x days (truck)",
        "Optimal x": 14,
        "x description": "every 14 days",
        "Avg Total Cost (€)": 25559.70,
        "Avg Storage Cost (€)": 11076.90,
        "Avg Transport Cost (€)": 14482.80,
        "Avg Shipments": 7.0,
        "Avg CO₂ Total (kg)": 13327.60,
        "Avg CO₂ Rail (kg)": 0.00,
        "Avg CO₂ Road (kg)": 13327.60,
        "Avg CO₂ per Shipment (kg)": 1903.90,
        "Avg CO₂ per kg Cargo (g)": 75.00,
    },
    {
        "Rule": "Rule 10: Fixed weekday(s) (truck)",
        "Optimal x": "Mo",
        "x description": "Mo",
        "Avg Total Cost (€)": 28139.94,
        "Avg Storage Cost (€)": 2568.04,
        "Avg Transport Cost (€)": 25571.90,
        "Avg Shipments": 13.0,
        "Avg CO₂ Total (kg)": 13327.60,
        "Avg CO₂ Rail (kg)": 0.00,
        "Avg CO₂ Road (kg)": 13327.60,
        "Avg CO₂ per Shipment (kg)": 1025.20,
        "Avg CO₂ per kg Cargo (g)": 75.00,
    },
    {
        "Rule": "Rule 11: kg threshold (truck)",
        "Optimal x": 7178.00,
        "x description": "7178.0 kg",
        "Avg Total Cost (€)": 36799.79,
        "Avg Storage Cost (€)": 1327.09,
        "Avg Transport Cost (€)": 35472.70,
        "Avg Shipments": 18.5,
        "Avg CO₂ Total (kg)": 13327.60,
        "Avg CO₂ Rail (kg)": 0.00,
        "Avg CO₂ Road (kg)": 13327.60,
        "Avg CO₂ per Shipment (kg)": 720.40,
        "Avg CO₂ per kg Cargo (g)": 75.00,
    },
    {
        "Rule": "Rule 12: cbm threshold (truck)",
        "Optimal x": 34.69,
        "x description": "34.7 cbm",
        "Avg Total Cost (€)": 38042.56,
        "Avg Storage Cost (€)": 1090.56,
        "Avg Transport Cost (€)": 36952.00,
        "Avg Shipments": 19.4,
        "Avg CO₂ Total (kg)": 13327.60,
        "Avg CO₂ Rail (kg)": 0.00,
        "Avg CO₂ Road (kg)": 13327.60,
        "Avg CO₂ per Shipment (kg)": 687.10,
        "Avg CO₂ per kg Cargo (g)": 75.00,
    },
]


def get_default_costs() -> Dict:
    return {
        "storage": {
            "free_days": 3,
            "tier1_days": (4, 7),
            "tier1_rate": 2.50,
            "tier2_day_from": 8,
            "tier2_rate": 4.00,
        },
        "intermodal": {
            "20GP": {"cost_eur": 1145.0, "max_kg": 20000, "max_cbm": 33.0},
            "40HQ": {"cost_eur": 1645.0, "max_kg": 24000, "max_cbm": 76.0},
        },
        "ltl": [
            (100, 146.0),
            (500, 285.0),
            (1000, 513.0),
            (2000, 616.0),
            (3000, 1069.0),
            (4000, 1494.0),
            (5000, 1522.0),
            (6000, 1782.0),
            (8000, 1878.0),
            (13000, 1956.0),
            (20000, 2085.0),
        ],
    }


def get_default_co2_params() -> Dict:
    return {
        "tare_20gp": 2.2,
        "tare_40hq": 4.0,
        "rail_km": 800,
        "road_leg_km": 100,
        "ltl_km": 1000,
        "ef_rail": 0.022,
        "ef_road": 0.075,
    }


DAY_COMBOS = [
    ("Tu",), ("We",), ("Th",), ("Fr",), ("Sa",), ("Su",), ("Mo",),
]


def clean_rule_name(rule_label: str) -> str:
    return rule_label.split(":")[0].strip()


def detect_mode(rule_label: str) -> str:
    if "intermodal" in rule_label:
        return "intermodal"
    if "truck" in rule_label:
        return "truck"
    return "mixed"


WEEKDAY_ALIASES = {
    "tu": "Tuesday",
    "tue": "Tuesday",
    "tues": "Tuesday",
    "tuesday": "Tuesday",
    "we": "Wednesday",
    "wed": "Wednesday",
    "weds": "Wednesday",
    "wednesday": "Wednesday",
    "th": "Thursday",
    "thu": "Thursday",
    "thur": "Thursday",
    "thurs": "Thursday",
    "thursday": "Thursday",
    "fr": "Friday",
    "fri": "Friday",
    "friday": "Friday",
    "sa": "Saturday",
    "sat": "Saturday",
    "saturday": "Saturday",
    "su": "Sunday",
    "sun": "Sunday",
    "sunday": "Sunday",
    "mo": "Monday",
    "mon": "Monday",
    "monday": "Monday",
}


@st.cache_data(show_spinner=False)
def load_local_database(path: str = "Database.csv") -> Optional[pd.DataFrame]:
    try:
        return read_historical_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def read_historical_csv(source) -> pd.DataFrame:
    if hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            text = raw.decode("utf-8-sig")
        else:
            text = raw
    else:
        with open(source, "rb") as f:
            text = f.read().decode("utf-8-sig")

    last_error = None
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
            df.columns = [str(c).strip() for c in df.columns]
            if EXPECTED_COLUMNS.issubset(df.columns):
                return normalize_historical_data(df)
        except Exception as exc:
            last_error = exc
    raise ValueError(
        "Could not read the historical data file. Expected columns: "
        "Weekday, Brutto Weight, Cbm."
    ) from last_error


def normalize_historical_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if not EXPECTED_COLUMNS.issubset(df.columns):
        raise ValueError("Historical data must include Weekday, Brutto Weight, and Cbm columns.")

    df = df[["Weekday", "Brutto Weight", "Cbm"]].copy()
    df["Weekday"] = (
        df["Weekday"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(WEEKDAY_ALIASES)
    )
    if df["Weekday"].isna().any():
        bad_values = sorted(df.loc[df["Weekday"].isna(), "Weekday"].astype(str).unique().tolist())
        raise ValueError(f"Unrecognized weekday values found in input data: {bad_values}")

    df["Brutto Weight"] = pd.to_numeric(df["Brutto Weight"], errors="coerce")
    df["Cbm"] = pd.to_numeric(df["Cbm"], errors="coerce")
    df = df.dropna(subset=["Brutto Weight", "Cbm"]).reset_index(drop=True)
    df["Brutto Weight"] = df["Brutto Weight"].clip(lower=0)
    df["Cbm"] = df["Cbm"].clip(lower=0)
    return df


def fit_model(df: pd.DataFrame) -> Dict:
    params = {"weekday_order": WEEKDAY_ORDER, "by_weekday": {}}

    for wd in WEEKDAY_ORDER:
        sub = df[df["Weekday"] == wd]
        n_tot = len(sub)
        nz = sub[(sub["Brutto Weight"] > 0) & (sub["Cbm"] > 0)]
        n_nz = len(nz)
        p_zero = 1.0 - n_nz / n_tot if n_tot > 0 else 0.0

        if n_nz >= 2:
            w_shape, _, w_scale = stats.gamma.fit(nz["Brutto Weight"], floc=0)
            c_shape, _, c_scale = stats.gamma.fit(nz["Cbm"], floc=0)
        else:
            all_nz = df[(df["Brutto Weight"] > 0) & (df["Cbm"] > 0)]
            if len(all_nz) < 2:
                raise ValueError(
                    "Not enough non-zero observations to fit the Gamma distributions. "
                    "Please provide more historical data."
                )
            w_shape, _, w_scale = stats.gamma.fit(all_nz["Brutto Weight"], floc=0)
            c_shape, _, c_scale = stats.gamma.fit(all_nz["Cbm"], floc=0)

        params["by_weekday"][wd] = {
            "p_zero": float(p_zero),
            "w_shape": float(w_shape),
            "w_scale": float(w_scale),
            "c_shape": float(c_shape),
            "c_scale": float(c_scale),
        }

    nz_all = df[(df["Brutto Weight"] > 0) & (df["Cbm"] > 0)].copy()
    if len(nz_all) >= 4:
        rho, _ = stats.spearmanr(np.log(nz_all["Brutto Weight"]), np.log(nz_all["Cbm"]))
        rho_pearson = 2 * np.sin(rho * math.pi / 6)
    else:
        rho_pearson = 0.95

    params["copula_rho"] = float(np.clip(rho_pearson, -0.9999, 0.9999))
    return params


def simulate(n_days: int, params: Dict, seed: Optional[int] = None) -> pd.DataFrame:
    if n_days < 1:
        raise ValueError("n_days must be at least 1")

    rng = np.random.default_rng(seed)
    rounded_days = math.ceil(n_days / 7) * 7
    rho = params["copula_rho"]
    cov = np.array([[1.0, rho], [rho, 1.0]])
    chol = np.linalg.cholesky(cov)

    rows = []
    for i in range(rounded_days):
        wd = params["weekday_order"][i % 7]
        p = params["by_weekday"][wd]
        day_no = i + 1

        if rng.random() < p["p_zero"]:
            weight = 0
            cbm = 0.0
        else:
            z = chol @ rng.standard_normal(2)
            u_w, u_c = stats.norm.cdf(z[0]), stats.norm.cdf(z[1])
            weight = stats.gamma.ppf(u_w, a=p["w_shape"], scale=p["w_scale"])
            cbm = stats.gamma.ppf(u_c, a=p["c_shape"], scale=p["c_scale"])
            weight = max(1, round(float(weight)))
            cbm = max(0.01, round(float(cbm), 2))

        rows.append(
            {
                "Day": day_no,
                "Weekday": WEEKDAY_SHORT_MAP[wd],
                "Brutto_Weight_kg": weight,
                "Cbm": cbm,
            }
        )
    return pd.DataFrame(rows)


def run_scenarios_from_params(params: Dict, n_simulations: int, n_days: int) -> List[pd.DataFrame]:
    return [simulate(n_days=n_days, params=params, seed=i) for i in range(n_simulations)]


class WarehouseState:
    def __init__(self, costs: Dict):
        self.lots: List[Dict[str, float]] = []
        self.costs = costs

    def arrive(self, kg: float, cbm: float) -> None:
        if kg > 0 or cbm > 0:
            self.lots.append({"kg": float(kg), "cbm": float(cbm), "age": 1})

    def accrue_storage_cost(self) -> float:
        s = self.costs["storage"]
        cost = 0.0
        new_lots = []
        for lot in self.lots:
            age = lot["age"]
            if age <= s["free_days"]:
                rate = 0.0
            elif age <= s["tier1_days"][1]:
                rate = s["tier1_rate"]
            else:
                rate = s["tier2_rate"]
            cost += lot["cbm"] * rate
            new_lots.append({"kg": lot["kg"], "cbm": lot["cbm"], "age": age + 1})
        self.lots = new_lots
        return cost

    def total_kg(self) -> float:
        return float(sum(l["kg"] for l in self.lots))

    def total_cbm(self) -> float:
        return float(sum(l["cbm"] for l in self.lots))

    def clear(self, kg_to_ship: float, cbm_to_ship: float) -> None:
        if kg_to_ship >= self.total_kg() - 0.01:
            self.lots = []
            return

        remaining_kg = kg_to_ship
        remaining_cbm = cbm_to_ship
        new_lots = []
        for lot in self.lots:
            if remaining_kg <= 0 and remaining_cbm <= 0:
                new_lots.append(lot)
                continue
            take_kg = min(lot["kg"], remaining_kg)
            take_cbm = min(lot["cbm"], remaining_cbm)
            remaining_kg -= take_kg
            remaining_cbm -= take_cbm
            leftover_kg = lot["kg"] - take_kg
            leftover_cbm = lot["cbm"] - take_cbm
            if leftover_kg > 0.01 or leftover_cbm > 0.01:
                new_lots.append({"kg": leftover_kg, "cbm": leftover_cbm, "age": lot["age"]})
        self.lots = new_lots


def ltl_cost(kg: float, costs: Dict) -> float:
    for max_kg, price in costs["ltl"]:
        if kg <= max_kg:
            return float(price)
    return float(costs["ltl"][-1][1])


def _fits_container(kg: float, cbm: float, container: Dict) -> bool:
    return kg <= container["max_kg"] and cbm <= container["max_cbm"]


def _intermodal_only_cost(kg: float, cbm: float, costs: Dict) -> float:
    c20 = costs["intermodal"]["20GP"]
    c40 = costs["intermodal"]["40HQ"]
    total_cost = 0.0
    rem_kg, rem_cbm = kg, cbm

    while rem_kg > 0.01 or rem_cbm > 0.01:
        if _fits_container(rem_kg, rem_cbm, c20):
            total_cost += min(c20["cost_eur"], c40["cost_eur"])
            break
        elif _fits_container(rem_kg, rem_cbm, c40):
            total_cost += c40["cost_eur"]
            break
        else:
            total_cost += c40["cost_eur"]
            rem_kg = max(0.0, rem_kg - c40["max_kg"])
            rem_cbm = max(0.0, rem_cbm - c40["max_cbm"])

    return float(total_cost)


def _mixed_cost(kg: float, cbm: float, costs: Dict) -> float:
    c20 = costs["intermodal"]["20GP"]
    c40 = costs["intermodal"]["40HQ"]

    best = ltl_cost(kg, costs)
    max_40 = int(np.ceil(kg / c40["max_kg"])) + 1
    max_20 = int(np.ceil(kg / c20["max_kg"])) + 1

    for n40 in range(0, max_40 + 1):
        for n20 in range(0, max_20 + 1):
            if n40 == 0 and n20 == 0:
                continue
            cap_kg = n40 * c40["max_kg"] + n20 * c20["max_kg"]
            cap_cbm = n40 * c40["max_cbm"] + n20 * c20["max_cbm"]
            container_cost = n40 * c40["cost_eur"] + n20 * c20["cost_eur"]
            if cap_kg >= kg and cap_cbm >= cbm:
                candidate = container_cost
            else:
                rem_kg = max(0.0, kg - cap_kg)
                candidate = container_cost + ltl_cost(rem_kg, costs)
            best = min(best, candidate)

    return float(best)


def best_transport_cost(kg: float, cbm: float, costs: Dict, mode: str = "mixed") -> float:
    if kg <= 0 and cbm <= 0:
        return 0.0
    if mode == "truck":
        return ltl_cost(kg, costs)
    if mode == "intermodal":
        return _intermodal_only_cost(kg, cbm, costs)
    return _mixed_cost(kg, cbm, costs)


def co2_ltl(cargo_kg: float, co2_params: Dict) -> Dict[str, float]:
    cargo_t = cargo_kg / 1000.0
    total = cargo_t * co2_params["ltl_km"] * co2_params["ef_road"]
    return {"co2_total_kg": total, "co2_rail_kg": 0.0, "co2_road_kg": total}


def co2_container(cargo_kg: float, container_type: str, co2_params: Dict) -> Dict[str, float]:
    tare_t = co2_params["tare_20gp"] if container_type == "20GP" else co2_params["tare_40hq"]
    gross_t = tare_t + cargo_kg / 1000.0
    rail = gross_t * co2_params["rail_km"] * co2_params["ef_rail"]
    road = gross_t * co2_params["road_leg_km"] * co2_params["ef_road"]
    return {"co2_total_kg": rail + road, "co2_rail_kg": rail, "co2_road_kg": road}


def best_transport_co2(kg: float, cbm: float, costs: Dict, co2_params: Dict, mode: str = "mixed") -> Dict[str, float]:
    if kg <= 0 and cbm <= 0:
        return {"co2_total_kg": 0.0, "co2_rail_kg": 0.0, "co2_road_kg": 0.0}

    c20 = costs["intermodal"]["20GP"]
    c40 = costs["intermodal"]["40HQ"]
    total_rail = 0.0
    total_road = 0.0

    if mode == "truck":
        return co2_ltl(kg, co2_params)

    if mode == "intermodal":
        rem_kg, rem_cbm = kg, cbm
        while rem_kg > 0.01 or rem_cbm > 0.01:
            if _fits_container(rem_kg, rem_cbm, c20):
                ctype = "20GP" if c20["cost_eur"] <= c40["cost_eur"] else "40HQ"
                em = co2_container(rem_kg, ctype, co2_params)
                total_rail += em["co2_rail_kg"]
                total_road += em["co2_road_kg"]
                break
            elif _fits_container(rem_kg, rem_cbm, c40):
                em = co2_container(rem_kg, "40HQ", co2_params)
                total_rail += em["co2_rail_kg"]
                total_road += em["co2_road_kg"]
                break
            else:
                cargo_this = min(rem_kg, c40["max_kg"])
                em = co2_container(cargo_this, "40HQ", co2_params)
                total_rail += em["co2_rail_kg"]
                total_road += em["co2_road_kg"]
                rem_kg = max(0.0, rem_kg - c40["max_kg"])
                rem_cbm = max(0.0, rem_cbm - c40["max_cbm"])
    else:
        best_cost = ltl_cost(kg, costs)
        best_n40 = 0
        best_n20 = 0
        best_rem = kg

        max_40 = int(np.ceil(kg / c40["max_kg"])) + 1
        max_20 = int(np.ceil(kg / c20["max_kg"])) + 1
        for n40 in range(0, max_40 + 1):
            for n20 in range(0, max_20 + 1):
                if n40 == 0 and n20 == 0:
                    continue
                cap_kg = n40 * c40["max_kg"] + n20 * c20["max_kg"]
                cap_cbm = n40 * c40["max_cbm"] + n20 * c20["max_cbm"]
                container_cost = n40 * c40["cost_eur"] + n20 * c20["cost_eur"]
                if cap_kg >= kg and cap_cbm >= cbm:
                    candidate = container_cost
                    rem = 0.0
                else:
                    rem = max(0.0, kg - cap_kg)
                    candidate = container_cost + ltl_cost(rem, costs)
                if candidate < best_cost:
                    best_cost = candidate
                    best_n40, best_n20, best_rem = n40, n20, rem

        remaining_kg = kg
        for _ in range(best_n40):
            cargo_this = min(remaining_kg, c40["max_kg"])
            em = co2_container(cargo_this, "40HQ", co2_params)
            total_rail += em["co2_rail_kg"]
            total_road += em["co2_road_kg"]
            remaining_kg = max(0.0, remaining_kg - c40["max_kg"])
        for _ in range(best_n20):
            cargo_this = min(remaining_kg, c20["max_kg"])
            em = co2_container(cargo_this, "20GP", co2_params)
            total_rail += em["co2_rail_kg"]
            total_road += em["co2_road_kg"]
            remaining_kg = max(0.0, remaining_kg - c20["max_kg"])
        if best_rem > 0.01:
            em = co2_ltl(best_rem, co2_params)
            total_road += em["co2_road_kg"]

    return {
        "co2_total_kg": total_rail + total_road,
        "co2_rail_kg": total_rail,
        "co2_road_kg": total_road,
    }


def simulate_policy(sim_df: pd.DataFrame, costs: Dict, dispatch_fn, co2_params: Dict, mode: str = "mixed") -> Dict[str, float]:
    wh = WarehouseState(costs)
    total_storage = 0.0
    total_transport = 0.0
    total_co2 = 0.0
    total_co2_rail = 0.0
    total_co2_road = 0.0
    total_cargo_kg = 0.0
    n_shipments = 0

    for _, row in sim_df.iterrows():
        day = int(row["Day"])
        weekday = row["Weekday"]
        kg = float(row["Brutto_Weight_kg"])
        cbm = float(row["Cbm"])

        wh.arrive(kg, cbm)
        total_storage += wh.accrue_storage_cost()

        if dispatch_fn(day, weekday, wh):
            ship_kg = wh.total_kg()
            ship_cbm = wh.total_cbm()
            if ship_kg > 0.01 or ship_cbm > 0.01:
                t_cost = best_transport_cost(ship_kg, ship_cbm, costs, mode)
                co2 = best_transport_co2(ship_kg, ship_cbm, costs, co2_params, mode)
                total_transport += t_cost
                total_co2 += co2["co2_total_kg"]
                total_co2_rail += co2["co2_rail_kg"]
                total_co2_road += co2["co2_road_kg"]
                total_cargo_kg += ship_kg
                n_shipments += 1
                wh.clear(ship_kg, ship_cbm)

    final_kg = wh.total_kg()
    final_cbm = wh.total_cbm()
    if final_kg > 0.01 or final_cbm > 0.01:
        t_cost = best_transport_cost(final_kg, final_cbm, costs, mode)
        co2 = best_transport_co2(final_kg, final_cbm, costs, co2_params, mode)
        total_transport += t_cost
        total_co2 += co2["co2_total_kg"]
        total_co2_rail += co2["co2_rail_kg"]
        total_co2_road += co2["co2_road_kg"]
        total_cargo_kg += final_kg
        n_shipments += 1

    co2_per_kg = total_co2 / total_cargo_kg if total_cargo_kg > 0 else 0.0
    return {
        "total_cost": total_storage + total_transport,
        "total_storage_cost": total_storage,
        "total_transport_cost": total_transport,
        "n_shipments": n_shipments,
        "days_simulated": len(sim_df),
        "co2_total_kg": total_co2,
        "co2_rail_kg": total_co2_rail,
        "co2_road_kg": total_co2_road,
        "co2_per_kg_cargo": co2_per_kg,
        "total_cargo_kg": total_cargo_kg,
    }


def avg_cost_across_sims(simulations: List[pd.DataFrame], costs: Dict, dispatch_fn, co2_params: Dict, mode: str = "mixed") -> float:
    totals = [simulate_policy(sim, costs, dispatch_fn, co2_params, mode)["total_cost"] for sim in simulations]
    return float(np.mean(totals))


def make_every_x_days(x: int):
    def dispatch(day, weekday, wh):
        return day % x == 0
    return dispatch


def make_fixed_weekdays(days: Tuple[str, ...]):
    days_set = set(days)
    def dispatch(day, weekday, wh):
        return weekday in days_set
    return dispatch


def make_kg_threshold(x: float):
    def dispatch(day, weekday, wh):
        return wh.total_kg() >= x
    return dispatch


def make_cbm_threshold(x: float):
    def dispatch(day, weekday, wh):
        return wh.total_cbm() >= x
    return dispatch


def _get_threshold_candidates(simulations: List[pd.DataFrame], key: str, n: int, extra_hi: Optional[float] = None) -> np.ndarray:
    vals = []
    for sim in simulations:
        vals.extend(sim[key].tolist())
    vals = [v for v in vals if v > 0]
    if not vals:
        lo, hi = 100, extra_hi or 10000
    else:
        lo = np.percentile(vals, 10)
        hi = np.percentile(vals, 95)
    if extra_hi is not None:
        hi = max(hi, extra_hi)
    return np.linspace(max(lo, 1), hi, n)


def optimize_all_rules(costs: Dict, simulations: List[pd.DataFrame], co2_params: Dict, day_candidates: int, kg_candidates_n: int, cbm_candidates_n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    raw_rows = []

    def evaluate(label: str, x_label: str, x_value, dispatch_fn, mode: str) -> Dict:
        detailed = [simulate_policy(sim, costs, dispatch_fn, co2_params, mode) for sim in simulations]
        for sim_idx, detail in enumerate(detailed, start=1):
            raw_rows.append(
                {
                    "Rule": label,
                    "Simulation": sim_idx,
                    "Mode": mode,
                    **detail,
                }
            )
        return {
            "Rule": label,
            "Optimal x": x_value,
            "x description": x_label,
            "Mode": mode,
            "Avg Total Cost (€)": round(np.mean([d["total_cost"] for d in detailed]), 2),
            "Avg Storage Cost (€)": round(np.mean([d["total_storage_cost"] for d in detailed]), 2),
            "Avg Transport Cost (€)": round(np.mean([d["total_transport_cost"] for d in detailed]), 2),
            "Avg Shipments": round(np.mean([d["n_shipments"] for d in detailed]), 1),
            "Avg CO₂ Total (kg)": round(np.mean([d["co2_total_kg"] for d in detailed]), 1),
            "Avg CO₂ Rail (kg)": round(np.mean([d["co2_rail_kg"] for d in detailed]), 1),
            "Avg CO₂ Road (kg)": round(np.mean([d["co2_road_kg"] for d in detailed]), 1),
            "Avg CO₂ per Shipment (kg)": round(
                np.mean([
                    d["co2_total_kg"] / d["n_shipments"] if d["n_shipments"] > 0 else 0
                    for d in detailed
                ]),
                1,
            ),
            "Avg CO₂ per kg Cargo (g)": round(np.mean([d["co2_per_kg_cargo"] * 1000 for d in detailed]), 2),
        }

    kg_candidates = _get_threshold_candidates(simulations, "Brutto_Weight_kg", kg_candidates_n)
    cbm_candidates = _get_threshold_candidates(simulations, "Cbm", cbm_candidates_n)
    kg_candidates_im = _get_threshold_candidates(
        simulations,
        "Brutto_Weight_kg",
        kg_candidates_n,
        extra_hi=costs["intermodal"]["40HQ"]["max_kg"] * 2,
    )
    cbm_candidates_im = _get_threshold_candidates(
        simulations,
        "Cbm",
        cbm_candidates_n,
        extra_hi=costs["intermodal"]["40HQ"]["max_cbm"] * 2,
    )

    def _opt_every_x(rule_label: str, mode: str) -> None:
        best_cost, best_x = np.inf, 1
        for x in range(1, day_candidates + 1):
            c = avg_cost_across_sims(simulations, costs, make_every_x_days(x), co2_params, mode)
            if c < best_cost:
                best_cost, best_x = c, x
        results.append(evaluate(rule_label, f"every {best_x} days", best_x, make_every_x_days(best_x), mode))

    def _opt_weekdays(rule_label: str, mode: str) -> None:
        best_cost, best_combo = np.inf, ("Mo",)
        for r in range(1, len(WEEKDAY_SHORT_MAP) + 1):
            for combo in __import__("itertools").combinations(WEEKDAY_SHORT_MAP.values(), r):
                c = avg_cost_across_sims(simulations, costs, make_fixed_weekdays(combo), co2_params, mode)
                if c < best_cost:
                    best_cost, best_combo = c, combo
        combo_str = " & ".join(best_combo)
        results.append(evaluate(rule_label, combo_str, combo_str, make_fixed_weekdays(best_combo), mode))

    def _opt_threshold(rule_label: str, candidates: np.ndarray, ttype: str, mode: str) -> None:
        best_cost, best_x = np.inf, float(candidates[0])
        for x in candidates:
            fn = make_kg_threshold(float(x)) if ttype == "kg" else make_cbm_threshold(float(x))
            c = avg_cost_across_sims(simulations, costs, fn, co2_params, mode)
            if c < best_cost:
                best_cost, best_x = c, float(x)
        fn = make_kg_threshold(best_x) if ttype == "kg" else make_cbm_threshold(best_x)
        unit = "kg" if ttype == "kg" else "cbm"
        results.append(evaluate(rule_label, f"{best_x:.1f} {unit}", round(best_x, 2), fn, mode))

    _opt_every_x("Rule 01: Every x days (mixed)", "mixed")
    _opt_weekdays("Rule 02: Fixed weekday(s) (mixed)", "mixed")
    _opt_threshold("Rule 03: kg threshold (mixed)", kg_candidates, "kg", "mixed")
    _opt_threshold("Rule 04: cbm threshold (mixed)", cbm_candidates, "cbm", "mixed")

    _opt_every_x("Rule 05: Every x days (intermodal)", "intermodal")
    _opt_weekdays("Rule 06: Fixed weekday(s) (intermodal)", "intermodal")
    _opt_threshold("Rule 07: kg threshold (intermodal)", kg_candidates_im, "kg", "intermodal")
    _opt_threshold("Rule 08: cbm threshold (intermodal)", cbm_candidates_im, "cbm", "intermodal")

    _opt_every_x("Rule 09: Every x days (truck)", "truck")
    _opt_weekdays("Rule 10: Fixed weekday(s) (truck)", "truck")
    _opt_threshold("Rule 11: kg threshold (truck)", kg_candidates, "kg", "truck")
    _opt_threshold("Rule 12: cbm threshold (truck)", cbm_candidates, "cbm", "truck")

    summary = pd.DataFrame(results)
    raw_df = pd.DataFrame(raw_rows)
    return summary, raw_df


@st.cache_data(show_spinner=False)
def run_pipeline_cached(df_json: str, costs_json: str, co2_json: str, settings_json: str):
    historical_df = pd.read_json(io.StringIO(df_json), orient="split")
    costs = json.loads(costs_json)
    costs["storage"]["tier1_days"] = tuple(costs["storage"]["tier1_days"])
    costs["ltl"] = [tuple(x) for x in costs["ltl"]]
    co2_params = json.loads(co2_json)
    settings = json.loads(settings_json)

    params = fit_model(historical_df)
    simulations = run_scenarios_from_params(params, settings["n_simulations"], settings["n_days"])
    summary, raw_df = optimize_all_rules(
        costs,
        simulations,
        co2_params,
        settings["day_candidates"],
        settings["kg_candidates"],
        settings["cbm_candidates"],
    )
    return {
        "params": params,
        "simulations": simulations,
        "summary": summary,
        "raw_df": raw_df,
    }


def build_weekday_fit_table(params: Dict) -> pd.DataFrame:
    rows = []
    for wd in WEEKDAY_ORDER:
        p = params["by_weekday"][wd]
        rows.append(
            {
                "Weekday": wd,
                "Zero-freight probability": round(p["p_zero"], 4),
                "Weight Gamma shape": round(p["w_shape"], 4),
                "Weight Gamma scale": round(p["w_scale"], 4),
                "CBM Gamma shape": round(p["c_shape"], 4),
                "CBM Gamma scale": round(p["c_scale"], 4),
            }
        )
    return pd.DataFrame(rows)


def build_historical_weekday_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wd in WEEKDAY_ORDER:
        sub = df[df["Weekday"] == wd]
        zero_mask = (sub["Brutto Weight"] <= 0) | (sub["Cbm"] <= 0)
        rows.append(
            {
                "Weekday": wd,
                "Observations": len(sub),
                "Zero-freight share": round(float(zero_mask.mean()) if len(sub) else 0.0, 4),
                "Avg Weight (kg)": round(float(sub["Brutto Weight"].mean()) if len(sub) else 0.0, 2),
                "Avg CBM": round(float(sub["Cbm"].mean()) if len(sub) else 0.0, 2),
            }
        )
    return pd.DataFrame(rows)


def add_comparison_columns(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    min_cost = summary["Avg Total Cost (€)"].min()
    min_co2 = summary["Avg CO₂ Total (kg)"].min()
    summary["Cost Gap vs Best (€)"] = (summary["Avg Total Cost (€)"] - min_cost).round(2)
    summary["CO₂ Gap vs Best (kg)"] = (summary["Avg CO₂ Total (kg)"] - min_co2).round(1)
    return summary.sort_values("Avg Total Cost (€)").reset_index(drop=True)


def compute_pareto_mask(summary: pd.DataFrame) -> List[bool]:
    costs_avg = summary["Avg Total Cost (€)"].values
    co2_avg = summary["Avg CO₂ Total (kg)"].values
    pareto_mask = np.ones(len(summary), dtype=bool)
    for i in range(len(summary)):
        for j in range(len(summary)):
            if i == j:
                continue
            if (
                costs_avg[j] <= costs_avg[i]
                and co2_avg[j] <= co2_avg[i]
                and (costs_avg[j] < costs_avg[i] or co2_avg[j] < co2_avg[i])
            ):
                pareto_mask[i] = False
                break
    return pareto_mask.tolist()


def create_pareto_chart(summary: pd.DataFrame) -> go.Figure:
    df = summary.copy()
    df["Mode"] = df["Rule"].apply(detect_mode)
    df["Rule Short"] = df["Rule"].apply(clean_rule_name)
    df["Pareto"] = compute_pareto_mask(df)

    fig = go.Figure()
    for mode in ["mixed", "intermodal", "truck"]:
        sub = df[df["Mode"] == mode]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["Avg CO₂ Total (kg)"],
                y=sub["Avg Total Cost (€)"],
                mode="markers+text",
                text=sub["Rule Short"],
                textposition="top center",
                name=mode.capitalize(),
                marker=dict(
                    size=np.where(sub["Pareto"], 18, 11),
                    symbol=["star" if v else "circle" for v in sub["Pareto"]],
                    color=MODE_COLORS[mode],
                    line=dict(color="white", width=1),
                ),
                customdata=np.stack([sub["Rule"], sub["x description"]], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Optimal x: %{customdata[1]}<br>"
                    "Avg CO₂: %{x:,.1f} kg<br>"
                    "Avg total cost: €%{y:,.2f}<extra></extra>"
                ),
            )
        )

    pareto_df = df[df["Pareto"]].sort_values("Avg CO₂ Total (kg)")
    if len(pareto_df) > 1:
        fig.add_trace(
            go.Scatter(
                x=pareto_df["Avg CO₂ Total (kg)"],
                y=pareto_df["Avg Total Cost (€)"],
                mode="lines",
                name="Pareto frontier",
                line=dict(color="#777777", dash="dash"),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="Pareto Frontier: Cost vs CO₂",
        xaxis_title="Average CO₂ (kg)",
        yaxis_title="Average Total Cost (€)",
        legend_title="Transport Mode",
        height=520,
    )
    return fig


def create_cost_breakdown_chart(summary: pd.DataFrame) -> go.Figure:
    df = summary.copy()
    df["Mode"] = df["Rule"].apply(detect_mode)
    df["Rule Short"] = df["Rule"].apply(clean_rule_name)
    base_colors = [MODE_COLORS[m] for m in df["Mode"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["Rule Short"],
            y=df["Avg Storage Cost (€)"],
            name="Storage",
            marker_color=[hex_to_rgba(c, 0.4) for c in base_colors],
            hovertemplate="%{x}<br>Storage: €%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["Rule Short"],
            y=df["Avg Transport Cost (€)"],
            name="Transport",
            marker_color=base_colors,
            hovertemplate="%{x}<br>Transport: €%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        barmode="stack",
        title="Cost Breakdown per Rule",
        xaxis_title="Rule",
        yaxis_title="Average Cost (€)",
        height=520,
    )
    return fig


def create_cost_distribution_chart(raw_df: pd.DataFrame) -> go.Figure:
    df = raw_df.copy()
    df["Rule Short"] = df["Rule"].apply(clean_rule_name)
    fig = px.box(
        df,
        x="Rule Short",
        y="total_cost",
        color="Mode",
        color_discrete_map=MODE_COLORS,
        points="outliers",
        title="Total Cost Distribution across Monte Carlo Simulations",
        labels={"total_cost": "Total Cost (€)", "Rule Short": "Rule"},
    )
    fig.update_layout(height=520, xaxis_title="Rule", yaxis_title="Total Cost (€)")
    return fig


def create_co2_breakdown_chart(summary: pd.DataFrame) -> go.Figure:
    df = summary.copy()
    df["Mode"] = df["Rule"].apply(detect_mode)
    df["Rule Short"] = df["Rule"].apply(clean_rule_name)
    base_colors = [MODE_COLORS[m] for m in df["Mode"]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=df["Rule Short"],
            y=df["Avg CO₂ Rail (kg)"],
            name="Rail CO₂",
            marker_color=[hex_to_rgba(c, 0.4) for c in base_colors],
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=df["Rule Short"],
            y=df["Avg CO₂ Road (kg)"],
            name="Road CO₂",
            marker_color=base_colors,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["Rule Short"],
            y=df["Avg CO₂ per kg Cargo (g)"],
            name="CO₂ per kg cargo",
            mode="lines+markers",
            line=dict(color="#C00000"),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        barmode="stack",
        title="CO₂ Breakdown: Rail vs Road per Rule",
        height=520,
    )
    fig.update_xaxes(title_text="Rule")
    fig.update_yaxes(title_text="Average CO₂ (kg)", secondary_y=False)
    fig.update_yaxes(title_text="CO₂ intensity (g / kg cargo)", secondary_y=True)
    return fig


def create_simulation_timeseries(sim_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=sim_df["Day"],
            y=sim_df["Brutto_Weight_kg"],
            name="Weight (kg)",
            hovertemplate="Day %{x}<br>Weight: %{y:,.0f} kg<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=sim_df["Day"],
            y=sim_df["Cbm"],
            name="CBM",
            mode="lines+markers",
            hovertemplate="Day %{x}<br>CBM: %{y:,.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(title="Example Simulated Daily Arrivals", height=460)
    fig.update_xaxes(title_text="Day")
    fig.update_yaxes(title_text="Weight (kg)", secondary_y=False)
    fig.update_yaxes(title_text="Volume (cbm)", secondary_y=True)
    return fig


def format_currency(value: float) -> str:
    return f"€{value:,.2f}"


def parse_ltl_breakpoints(text: str) -> List[Tuple[int, float]]:
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid LTL row: {line}")
        max_kg = int(float(parts[0]))
        price = float(parts[1])
        rows.append((max_kg, price))
    rows = sorted(rows, key=lambda x: x[0])
    if not rows:
        raise ValueError("Please provide at least one LTL breakpoint.")
    return rows


def default_ltl_text(costs: Dict) -> str:
    return "\n".join(f"{kg},{price:g}" for kg, price in costs["ltl"])


def render_project_overview() -> None:
    st.title("📦 Heinz Logistics — Wait or Ship?")
    st.markdown(
        "A Monte Carlo decision-support dashboard for deciding **when to dispatch freight** "
        "from the warehouse and **which transport mode** to use."
    )
    st.info(
        "The notebook combines three components: a freight arrival simulator, a warehouse/storage cost model, "
        "and a rule optimizer that compares mixed, intermodal-only, and truck-only dispatch policies."
    )


def render_methodology() -> None:
    with st.expander("Methodology / Workflow", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.markdown(
            "**1. Fit the arrival model**\n\n"
            "Historical daily freight data is split by weekday. For each weekday, the model estimates "
            "the probability of no freight and fits Gamma distributions for non-zero weight and volume."
        )
        col2.markdown(
            "**2. Simulate future arrivals**\n\n"
            "Daily weight and cbm are generated with a Gaussian copula so the historical dependency between "
            "weight and volume is preserved instead of using a fixed kg/cbm ratio."
        )
        col3.markdown(
            "**3. Optimize dispatch rules**\n\n"
            "For each rule family and each candidate value of x, the app simulates warehouse storage, transport cost, "
            "and CO₂, then selects the x with the lowest average total cost."
        )
        st.markdown(
            "**Rule families tested:** every x days, fixed weekday(s), kg threshold, cbm threshold.  \\\n"
            "**Transport modes tested:** mixed, intermodal-only, truck-only.  \\\n"
            "**Optimization target:** average total cost.  \\\n"
            "**Reported side output:** CO₂ split into rail and road."
        )


def render_sidebar_inputs(local_df_available: bool):
    st.sidebar.header("Inputs")
    uploaded_file = st.sidebar.file_uploader("Upload historical freight CSV", type=["csv"])
    use_local = False
    if local_df_available:
        use_local = st.sidebar.checkbox("Use local Database.csv if no upload is provided", value=True)

    st.sidebar.subheader("Simulation Settings")
    n_simulations = st.sidebar.slider("Monte Carlo simulations", min_value=1, max_value=100, value=10, step=1)
    n_days = st.sidebar.slider("Days per simulation", min_value=7, max_value=365, value=90, step=1)

    st.sidebar.subheader("Grid Search Resolution")
    day_candidates = st.sidebar.slider("Every-x day candidates", min_value=1, max_value=60, value=30, step=1)
    kg_candidates = st.sidebar.slider("KG threshold candidates", min_value=5, max_value=60, value=30, step=1)
    cbm_candidates = st.sidebar.slider("CBM threshold candidates", min_value=5, max_value=60, value=30, step=1)

    defaults = get_default_costs()
    st.sidebar.subheader("Storage Costs")
    free_days = st.sidebar.number_input("Free storage days", min_value=0, value=defaults["storage"]["free_days"], step=1)
    tier1_last_day = st.sidebar.number_input("Tier 1 last day", min_value=int(free_days), value=defaults["storage"]["tier1_days"][1], step=1)
    tier1_rate = st.sidebar.number_input("Tier 1 rate (€ / wm / day)", min_value=0.0, value=float(defaults["storage"]["tier1_rate"]), step=0.1)
    tier2_rate = st.sidebar.number_input("Tier 2 rate (€ / wm / day)", min_value=0.0, value=float(defaults["storage"]["tier2_rate"]), step=0.1)

    st.sidebar.subheader("Intermodal Costs and Capacities")
    cost_20 = st.sidebar.number_input("20GP cost (€)", min_value=0.0, value=float(defaults["intermodal"]["20GP"]["cost_eur"]), step=10.0)
    kg_20 = st.sidebar.number_input("20GP max kg", min_value=1, value=int(defaults["intermodal"]["20GP"]["max_kg"]), step=100)
    cbm_20 = st.sidebar.number_input("20GP max cbm", min_value=0.1, value=float(defaults["intermodal"]["20GP"]["max_cbm"]), step=0.5)
    cost_40 = st.sidebar.number_input("40HQ cost (€)", min_value=0.0, value=float(defaults["intermodal"]["40HQ"]["cost_eur"]), step=10.0)
    kg_40 = st.sidebar.number_input("40HQ max kg", min_value=1, value=int(defaults["intermodal"]["40HQ"]["max_kg"]), step=100)
    cbm_40 = st.sidebar.number_input("40HQ max cbm", min_value=0.1, value=float(defaults["intermodal"]["40HQ"]["max_cbm"]), step=0.5)

    st.sidebar.subheader("Truck LTL Breakpoints")
    ltl_text = st.sidebar.text_area(
        "One row per breakpoint: max_kg,price",
        value=default_ltl_text(defaults),
        height=220,
    )

    co2_defaults = get_default_co2_params()
    with st.sidebar.expander("CO₂ Assumptions", expanded=False):
        tare_20gp = st.number_input("20GP tare (t)", min_value=0.0, value=float(co2_defaults["tare_20gp"]), step=0.1)
        tare_40hq = st.number_input("40HQ tare (t)", min_value=0.0, value=float(co2_defaults["tare_40hq"]), step=0.1)
        rail_km = st.number_input("Rail distance (km)", min_value=0, value=int(co2_defaults["rail_km"]), step=10)
        road_leg_km = st.number_input("Intermodal road leg distance (km)", min_value=0, value=int(co2_defaults["road_leg_km"]), step=10)
        ltl_km = st.number_input("LTL road distance (km)", min_value=0, value=int(co2_defaults["ltl_km"]), step=10)
        ef_rail = st.number_input("Rail emission factor (kg CO₂e / t·km)", min_value=0.0, value=float(co2_defaults["ef_rail"]), step=0.001, format="%.3f")
        ef_road = st.number_input("Road emission factor (kg CO₂e / t·km)", min_value=0.0, value=float(co2_defaults["ef_road"]), step=0.001, format="%.3f")

    costs = {
        "storage": {
            "free_days": int(free_days),
            "tier1_days": (int(free_days) + 1, int(tier1_last_day)),
            "tier1_rate": float(tier1_rate),
            "tier2_day_from": int(tier1_last_day) + 1,
            "tier2_rate": float(tier2_rate),
        },
        "intermodal": {
            "20GP": {"cost_eur": float(cost_20), "max_kg": int(kg_20), "max_cbm": float(cbm_20)},
            "40HQ": {"cost_eur": float(cost_40), "max_kg": int(kg_40), "max_cbm": float(cbm_40)},
        },
    }
    costs["ltl"] = parse_ltl_breakpoints(ltl_text)

    co2_params = {
        "tare_20gp": float(tare_20gp),
        "tare_40hq": float(tare_40hq),
        "rail_km": int(rail_km),
        "road_leg_km": int(road_leg_km),
        "ltl_km": int(ltl_km),
        "ef_rail": float(ef_rail),
        "ef_road": float(ef_road),
    }

    settings = {
        "n_simulations": int(n_simulations),
        "n_days": int(n_days),
        "day_candidates": int(day_candidates),
        "kg_candidates": int(kg_candidates),
        "cbm_candidates": int(cbm_candidates),
    }
    return uploaded_file, use_local, costs, co2_params, settings


def render_kpis(summary: pd.DataFrame) -> None:
    cheapest = summary.loc[summary["Avg Total Cost (€)"].idxmin()]
    greenest = summary.loc[summary["Avg CO₂ Total (kg)"].idxmin()]
    best_intermodal = summary[summary["Mode"] == "intermodal"].sort_values("Avg Total Cost (€)").iloc[0]
    best_truck = summary[summary["Mode"] == "truck"].sort_values("Avg Total Cost (€)").iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lowest Cost Rule", clean_rule_name(cheapest["Rule"]), format_currency(cheapest["Avg Total Cost (€)"]))
    c2.metric("Lowest CO₂ Rule", clean_rule_name(greenest["Rule"]), f"{greenest['Avg CO₂ Total (kg)']:,.1f} kg")
    c3.metric("Best Intermodal Cost", clean_rule_name(best_intermodal["Rule"]), format_currency(best_intermodal["Avg Total Cost (€)"]))
    c4.metric("Best Truck Cost", clean_rule_name(best_truck["Rule"]), format_currency(best_truck["Avg Total Cost (€)"]))


def render_snapshot_mode() -> None:
    st.warning(
        "`Database.csv` was not attached with the notebook, so the app cannot rerun the Monte Carlo optimization yet. "
        "Upload the historical CSV to activate the full interactive model."
    )
    snapshot = pd.DataFrame(NOTEBOOK_SUMMARY_ROWS)
    snapshot["Mode"] = snapshot["Rule"].apply(detect_mode)
    render_kpis(snapshot)
    st.subheader("Notebook Result Snapshot")
    st.dataframe(add_comparison_columns(snapshot), use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Pareto", "Cost Breakdown", "CO₂ Breakdown"])
    with tab1:
        st.plotly_chart(create_pareto_chart(snapshot), use_container_width=True)
    with tab2:
        st.plotly_chart(create_cost_breakdown_chart(snapshot), use_container_width=True)
    with tab3:
        st.plotly_chart(create_co2_breakdown_chart(snapshot), use_container_width=True)

    st.caption(
        "This snapshot reproduces the summary table and the main insights from the notebook run shown in the uploaded file. "
        "The cost-distribution box plot is only available after rerunning with the original historical data."
    )


def render_active_model(historical_df: pd.DataFrame, pipeline_results: Dict, settings: Dict) -> None:
    params = pipeline_results["params"]
    simulations = pipeline_results["simulations"]
    summary = pipeline_results["summary"].copy()
    raw_df = pipeline_results["raw_df"].copy()

    summary["Mode"] = summary["Rule"].apply(detect_mode)
    summary_with_gaps = add_comparison_columns(summary)
    render_kpis(summary)

    tabs = st.tabs([
        "Results",
        "Visualizations",
        "Historical Data & Model Fit",
        "Simulation Preview",
    ])

    with tabs[0]:
        st.subheader("Optimization Results")
        st.dataframe(summary_with_gaps, use_container_width=True)

        csv_bytes = summary_with_gaps.to_csv(index=False).encode("utf-8")
        st.download_button("Download summary CSV", data=csv_bytes, file_name="heinz_logistics_summary.csv", mime="text/csv")

        cheapest = summary.loc[summary["Avg Total Cost (€)"].idxmin()]
        greenest = summary.loc[summary["Avg CO₂ Total (kg)"].idxmin()]
        st.markdown(
            f"**Best cost rule:** {cheapest['Rule']} with average total cost of {format_currency(cheapest['Avg Total Cost (€)'])}.  \\\n"
            f"**Best CO₂ rule:** {greenest['Rule']} with average CO₂ of {greenest['Avg CO₂ Total (kg)']:,.1f} kg."
        )

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(create_pareto_chart(summary), use_container_width=True)
            st.plotly_chart(create_cost_distribution_chart(raw_df), use_container_width=True)
        with c2:
            st.plotly_chart(create_cost_breakdown_chart(summary), use_container_width=True)
            st.plotly_chart(create_co2_breakdown_chart(summary), use_container_width=True)

    with tabs[2]:
        left, right = st.columns([1.1, 1])
        with left:
            st.subheader("Uploaded Historical Data")
            st.dataframe(historical_df.head(20), use_container_width=True)
            st.caption(f"Rows: {len(historical_df):,} | Distinct weekdays: {historical_df['Weekday'].nunique()}")
            st.subheader("Weekday Summary")
            st.dataframe(build_historical_weekday_summary(historical_df), use_container_width=True)
        with right:
            st.subheader("Fitted Simulation Parameters")
            st.metric("Gaussian copula correlation", f"{params['copula_rho']:.4f}")
            st.dataframe(build_weekday_fit_table(params), use_container_width=True)

    with tabs[3]:
        st.subheader("Monte Carlo Scenario Preview")
        sim_index = st.slider("Choose simulation run", min_value=1, max_value=len(simulations), value=1, step=1)
        sim_df = simulations[sim_index - 1]
        st.caption(
            f"The notebook rounds {settings['n_days']} requested days up to the next full Tuesday–Monday cycle, "
            f"so each simulation shown here contains {len(sim_df)} days."
        )
        st.plotly_chart(create_simulation_timeseries(sim_df), use_container_width=True)
        st.dataframe(sim_df.head(30), use_container_width=True)

        weekday_agg = sim_df.groupby("Weekday", as_index=False).agg(
            Total_Weight_kg=("Brutto_Weight_kg", "sum"),
            Total_Cbm=("Cbm", "sum"),
            Zero_Days=("Brutto_Weight_kg", lambda s: int((s == 0).sum())),
        )
        weekday_agg["Weekday"] = pd.Categorical(weekday_agg["Weekday"], categories=list(WEEKDAY_SHORT_MAP.values()), ordered=True)
        weekday_agg = weekday_agg.sort_values("Weekday")
        st.dataframe(weekday_agg, use_container_width=True)


def main() -> None:
    render_project_overview()
    render_methodology()

    local_df = load_local_database()

    try:
        uploaded_file, use_local, costs, co2_params, settings = render_sidebar_inputs(local_df is not None)
    except ValueError as exc:
        st.sidebar.error(str(exc))
        st.stop()

    historical_df = None
    if uploaded_file is not None:
        try:
            historical_df = read_historical_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Failed to read uploaded CSV: {exc}")
            st.stop()
    elif use_local and local_df is not None:
        historical_df = local_df.copy()

    if historical_df is None:
        render_snapshot_mode()
        return

    df_json = historical_df.to_json(orient="split")
    costs_json = json.dumps(costs, sort_keys=True)
    co2_json = json.dumps(co2_params, sort_keys=True)
    settings_json = json.dumps(settings, sort_keys=True)

    with st.spinner("Running Monte Carlo simulation and rule optimization..."):
        try:
            pipeline_results = run_pipeline_cached(df_json, costs_json, co2_json, settings_json)
        except Exception as exc:
            st.error(f"Model execution failed: {exc}")
            st.stop()

    render_active_model(historical_df, pipeline_results, settings)


if __name__ == "__main__":
    main()
