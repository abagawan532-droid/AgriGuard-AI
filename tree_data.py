import json
import os
import re
import math

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tree_species.json")

def all_trees():
    if not os.path.exists(DATA_PATH):
        return []
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)

def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

def search_trees(query="", category="all"):
    data = all_trees()
    q = query.lower().strip()
    results = []
    for t in data:
        if category and category != "all" and t.get("category", "") != category:
            continue
        if q:
            match = (
                q in t.get("common_name", "").lower() or
                q in t.get("scientific_name", "").lower() or
                q in t.get("family", "").lower() or
                q in t.get("distribution", "").lower() or
                q in " ".join(t.get("other_names", [])).lower()
            )
            if not match:
                continue
        results.append(t)
    return results

def get_tree(key):
    if not key:
        return None
    key = str(key).lower().strip()
    trees = all_trees()
    for t in trees:
        if key in (t.get("id", "").lower(), _slug(t.get("scientific_name", "")), _slug(t.get("common_name", ""))):
            return t
    for t in trees:
        if key == t.get("scientific_name", "").lower() or key == t.get("common_name", "").lower():
            return t
    for t in trees:
        if key in t.get("common_name", "").lower() or key in t.get("id", "").lower():
            return t
    return None

def calculate_tree_value(unit_price, quantity, costs):
    if min(unit_price, quantity, costs) < 0:
        raise ValueError("Values must be non-negative")
    gross = float(unit_price) * float(quantity)
    return {
        "gross": round(gross, 2),
        "cost": round(float(costs), 2),
        "net": round(gross - float(costs), 2)
    }

def calculate_forestry_biomass(dbh_cm, height_m, species_id="teak", form_factor=0.65):
    """
    Computes standing tree timber volume (m³ and cu. ft) using Huber's Forestry formula,
    biomass, annual & total carbon sequestration (kg CO2), and commercial timber value.
    """
    dbh = float(dbh_cm)
    height = float(height_m)
    if dbh <= 0 or height <= 0:
        raise ValueError("Dimensions must be positive")
    
    radius_m = (dbh / 2.0) / 100.0  # cm to m
    basal_area = math.pi * (radius_m ** 2)
    volume_m3 = round(basal_area * height * form_factor, 3)
    volume_cuft = round(volume_m3 * 35.3147, 2)
    
    tree_info = get_tree(species_id) or {}
    unit_price_cuft = float(tree_info.get("default_unit_price", 3000))
    annual_carbon_kg = float(tree_info.get("carbon_rate_kg_yr", 25.0))
    
    # Biomass density ~ 650 kg/m³
    wood_density = 650.0
    biomass_kg = round(volume_m3 * wood_density, 1)
    # Carbon content ~ 50% of biomass, CO2 equivalent = Carbon * 44/12 (~ 3.67)
    total_co2_sequestered_kg = round(biomass_kg * 0.5 * (44.0 / 12.0), 1)
    
    gross_timber_val = round(volume_cuft * unit_price_cuft, 2)
    estimated_harvest_cost = round(gross_timber_val * 0.15, 2)
    net_timber_val = round(gross_timber_val - estimated_harvest_cost, 2)
    
    # Voluntary carbon market credit estimate (~ $10 / ₹830 per tonne CO2)
    carbon_credit_value = round((total_co2_sequestered_kg / 1000.0) * 850.0, 2)
    
    return {
        "species": tree_info.get("common_name", species_id.capitalize()),
        "dbh_cm": dbh,
        "height_m": height,
        "volume_m3": volume_m3,
        "volume_cuft": volume_cuft,
        "biomass_kg": biomass_kg,
        "total_co2_kg": total_co2_sequestered_kg,
        "annual_co2_kg": annual_carbon_kg,
        "unit_price_cuft": unit_price_cuft,
        "gross_timber_val": gross_timber_val,
        "harvest_cost": estimated_harvest_cost,
        "net_timber_val": net_timber_val,
        "carbon_credit_value": carbon_credit_value
    }
