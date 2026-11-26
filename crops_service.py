import json
import os
import re

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "crops_catalog.json")

def _load_data():
    if not os.path.exists(DATA_PATH):
        return {"total_crops": 0, "categories": [], "crops": []}
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_crop_catalog():
    return _load_data()

def get_all_crops():
    return _load_data().get("crops", [])

def get_categories():
    return _load_data().get("categories", [])

def _slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

def get_crop_by_id(crop_id):
    crop_id = crop_id.lower().strip()
    crops = get_all_crops()
    for c in crops:
        if c.get("id", "").lower() == crop_id or _slugify(c.get("name", "")) == crop_id:
            return c
    for c in crops:
        if crop_id in c.get("name", "").lower():
            return c
    return None

def search_crops(query="", category=None, season=None, water=None):
    crops = get_all_crops()
    q = query.lower().strip()
    results = []
    
    for c in crops:
        # Category filter
        if category and category != "all" and c.get("cat") != category:
            continue
        # Season filter
        if season and season != "all" and season.lower() not in c.get("season", "").lower():
            continue
        # Water filter
        if water and water != "all" and water.lower() not in c.get("water", "").lower():
            continue
        # Search query filter
        if q:
            match = (
                q in c.get("name", "").lower() or
                q in c.get("scientific", "").lower() or
                q in c.get("cat_name", "").lower() or
                any(q in d.lower() for d in c.get("diseases", [])) or
                any(q in p.lower() for p in c.get("pests", [])) or
                q in c.get("soil", "").lower()
            )
            if not match:
                continue
        results.append(c)
        
    return results

def calculate_crop_economics(crop_id, acreage=1.0, custom_yield=None, custom_price=None, custom_cost=None):
    crop = get_crop_by_id(crop_id)
    if not crop:
        return None
    
    # Extract baseline yield in quintals from string e.g. "20-28 quintals"
    base_yield = 15.0
    yield_str = crop.get("yield_acre", "")
    match = re.search(r"(\d+)(?:\s*-\s*(\d+))?", yield_str)
    if match:
        min_y = float(match.group(1))
        max_y = float(match.group(2)) if match.group(2) else min_y
        base_yield = (min_y + max_y) / 2.0
    
    base_price = float(crop.get("price_quintal", 3000))
    # Standard cultivation cost benchmark: approx 35-45% of gross revenue
    base_cost_per_acre = round(base_yield * base_price * 0.40, 2)
    
    actual_yield = float(custom_yield) if custom_yield is not None else base_yield
    actual_price = float(custom_price) if custom_price is not None else base_price
    actual_cost = float(custom_cost) if custom_cost is not None else base_cost_per_acre
    
    total_yield = round(actual_yield * acreage, 2)
    gross_revenue = round(total_yield * actual_price, 2)
    total_cost = round(actual_cost * acreage, 2)
    net_profit = round(gross_revenue - total_cost, 2)
    roi_percent = round((net_profit / total_cost * 100), 1) if total_cost > 0 else 0.0
    
    return {
        "crop": crop,
        "acreage": acreage,
        "yield_per_acre": actual_yield,
        "total_yield": total_yield,
        "price_per_quintal": actual_price,
        "cost_per_acre": actual_cost,
        "total_cost": total_cost,
        "gross_revenue": gross_revenue,
        "net_profit": net_profit,
        "roi_percent": roi_percent
    }
