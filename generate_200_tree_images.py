"""Generates high-resolution botanical vector artwork for all 200 tree species.
Saves to static/images/trees/<tree_id>.svg.
Each SVG includes custom canopy geometry, trunk styling, seasonal foliage accents,
and botanical metadata.
"""
import os
import json
import html

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPECIES_FILE = os.path.join(BASE_DIR, "data", "tree_species.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "images", "trees")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Category styling palettes
PALETTES = {
    "timber": {
        "bg_start": "#1c1917", "bg_end": "#292524", "accent": "#b45309",
        "leaf_col1": "#15803d", "leaf_col2": "#166534", "trunk": "#78350f"
    },
    "fruit": {
        "bg_start": "#431407", "bg_end": "#7c2d12", "accent": "#ea580c",
        "leaf_col1": "#65a30d", "leaf_col2": "#4d7c0f", "trunk": "#854d0e"
    },
    "medicinal": {
        "bg_start": "#022c22", "bg_end": "#064e3b", "accent": "#10b981",
        "leaf_col1": "#10b981", "leaf_col2": "#047857", "trunk": "#3f3f46"
    },
    "sacred": {
        "bg_start": "#1e1b4b", "bg_end": "#312e81", "accent": "#8b5cf6",
        "leaf_col1": "#22c55e", "leaf_col2": "#15803d", "trunk": "#52525b"
    },
    "plantation": {
        "bg_start": "#064e3b", "bg_end": "#065f46", "accent": "#14b8a6",
        "leaf_col1": "#14b8a6", "leaf_col2": "#0f766e", "trunk": "#713f12"
    },
    "ornamental": {
        "bg_start": "#4a044e", "bg_end": "#701a75", "accent": "#ec4899",
        "leaf_col1": "#ec4899", "leaf_col2": "#a21caf", "trunk": "#581c87"
    }
}

DEFAULT_PALETTE = {
    "bg_start": "#0f172a", "bg_end": "#1e293b", "accent": "#10b981",
    "leaf_col1": "#16a34a", "leaf_col2": "#15803d", "trunk": "#78350f"
}

def generate_svg(tree):
    tid = tree.get("id", "tree")
    name = html.escape(tree.get("common_name", "Tree Species"))
    sci = html.escape(tree.get("scientific_name", ""))
    cat = tree.get("category", "timber").lower()
    co2 = tree.get("carbon_rate_kg_yr", 25.0)
    emoji = tree.get("icon_emoji", "🌳")
    family = html.escape(tree.get("family", "Botanical Family"))

    p = PALETTES.get(cat, DEFAULT_PALETTE)

    # Different canopy silhouettes based on category or tree id
    is_palm = any(k in tid for k in ["coconut", "areca", "palm", "date"])
    is_banyan = "banyan" in tid or "peepal" in tid or "ficus" in tid

    if is_palm:
        # Palm canopy
        canopy_svg = f"""
        <!-- Palm Fronds -->
        <path d="M 200 170 Q 140 130 90 160 Q 140 150 200 170" fill="{p['leaf_col1']}" opacity="0.9"/>
        <path d="M 200 170 Q 260 130 310 160 Q 260 150 200 170" fill="{p['leaf_col1']}" opacity="0.9"/>
        <path d="M 200 170 Q 110 90 70 120 Q 120 120 200 170" fill="{p['leaf_col2']}" opacity="0.95"/>
        <path d="M 200 170 Q 290 90 330 120 Q 280 120 200 170" fill="{p['leaf_col2']}" opacity="0.95"/>
        <path d="M 200 170 Q 200 60 190 40 Q 210 60 200 170" fill="{p['leaf_col1']}"/>
        <!-- Ringed Trunk -->
        <path d="M 194 170 C 190 230 188 300 192 360 L 208 360 C 212 300 210 230 206 170 Z" fill="{p['trunk']}"/>
        <line x1="190" y1="210" x2="210" y2="210" stroke="#fef08a" stroke-width="1.5" opacity="0.4"/>
        <line x1="189" y1="250" x2="211" y2="250" stroke="#fef08a" stroke-width="1.5" opacity="0.4"/>
        <line x1="188" y1="290" x2="212" y2="290" stroke="#fef08a" stroke-width="1.5" opacity="0.4"/>
        <line x1="190" y1="330" x2="210" y2="330" stroke="#fef08a" stroke-width="1.5" opacity="0.4"/>
        """
    elif is_banyan:
        # Spreading massive canopy with aerial prop roots
        canopy_svg = f"""
        <!-- Aerial Prop Roots -->
        <line x1="120" y1="210" x2="115" y2="360" stroke="{p['trunk']}" stroke-width="4" stroke-linecap="round"/>
        <line x1="150" y1="220" x2="148" y2="360" stroke="{p['trunk']}" stroke-width="3"/>
        <line x1="280" y1="210" x2="285" y2="360" stroke="{p['trunk']}" stroke-width="4" stroke-linecap="round"/>
        <line x1="250" y1="220" x2="252" y2="360" stroke="{p['trunk']}" stroke-width="3"/>
        <!-- Central Trunk -->
        <path d="M 180 230 Q 185 300 170 360 L 230 360 Q 215 300 220 230 Z" fill="{p['trunk']}"/>
        <!-- Massive Spreading Canopy -->
        <ellipse cx="200" cy="150" rx="140" ry="85" fill="{p['leaf_col2']}" opacity="0.95"/>
        <ellipse cx="140" cy="140" rx="70" ry="55" fill="{p['leaf_col1']}" opacity="0.9"/>
        <ellipse cx="260" cy="140" rx="70" ry="55" fill="{p['leaf_col1']}" opacity="0.9"/>
        <ellipse cx="200" cy="115" rx="90" ry="60" fill="{p['leaf_col1']}"/>
        """
    else:
        # Standard mature forest & orchard canopy
        canopy_svg = f"""
        <!-- Trunk & Major Branches -->
        <path d="M 188 220 C 188 270 184 320 180 360 L 220 360 C 216 320 212 270 212 220 Z" fill="{p['trunk']}"/>
        <path d="M 194 230 Q 160 190 140 170" stroke="{p['trunk']}" stroke-width="8" stroke-linecap="round"/>
        <path d="M 206 230 Q 240 190 260 170" stroke="{p['trunk']}" stroke-width="8" stroke-linecap="round"/>
        <!-- Lush Foliage Layers -->
        <ellipse cx="200" cy="150" rx="110" ry="85" fill="{p['leaf_col2']}" opacity="0.9"/>
        <circle cx="145" cy="150" r="60" fill="{p['leaf_col1']}" opacity="0.95"/>
        <circle cx="255" cy="150" r="60" fill="{p['leaf_col1']}" opacity="0.95"/>
        <circle cx="200" cy="110" r="70" fill="{p['leaf_col1']}"/>
        <circle cx="170" cy="95" r="45" fill="{p['leaf_col1']}" opacity="0.8"/>
        <circle cx="230" cy="95" r="45" fill="{p['leaf_col1']}" opacity="0.8"/>
        """

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 480" width="100%" height="100%">
  <defs>
    <linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{p['bg_start']}"/>
      <stop offset="100%" stop-color="{p['bg_end']}"/>
    </linearGradient>
    <linearGradient id="soilGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#3f3f46"/>
      <stop offset="100%" stop-color="#18181b"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#000" flood-opacity="0.3"/>
    </filter>
  </defs>

  <!-- Background Canvas -->
  <rect width="400" height="480" rx="16" fill="url(#bgGrad)"/>
  
  <!-- Subtle Sunlight Halo -->
  <circle cx="200" cy="130" r="140" fill="{p['accent']}" opacity="0.12" filter="url(#shadow)"/>

  <!-- Soil Ground Bed -->
  <ellipse cx="200" cy="370" rx="170" ry="30" fill="url(#soilGrad)"/>
  <ellipse cx="200" cy="365" rx="150" ry="12" fill="#15803d" opacity="0.35"/>

  <!-- Botanical Tree Geometry -->
  <g filter="url(#shadow)">
    {canopy_svg}
  </g>

  <!-- Category & Species Header Pill -->
  <g transform="translate(18, 20)">
    <rect width="130" height="26" rx="13" fill="rgba(255,255,255,0.12)" stroke="rgba(255,255,255,0.2)"/>
    <text x="65" y="17" fill="#ffffff" font-size="11" font-weight="700" font-family="system-ui, -apple-system, sans-serif" text-anchor="middle">
      {cat.upper()} • #{tree.get("num", 1)}
    </text>
  </g>

  <!-- Carbon Absorption Badge -->
  <g transform="translate(262, 20)">
    <rect width="120" height="26" rx="13" fill="rgba(16, 185, 129, 0.2)" stroke="#10b981"/>
    <text x="60" y="17" fill="#34d399" font-size="11" font-weight="700" font-family="system-ui, -apple-system, sans-serif" text-anchor="middle">
      🌱 {co2} kg CO₂/yr
    </text>
  </g>

  <!-- Bottom Botanical Label Panel -->
  <rect x="16" y="390" width="368" height="74" rx="12" fill="rgba(15, 23, 42, 0.85)" stroke="rgba(255,255,255,0.1)"/>
  
  <text x="32" y="418" fill="#ffffff" font-size="18" font-weight="800" font-family="system-ui, -apple-system, sans-serif">
    {emoji} {name}
  </text>
  
  <text x="32" y="438" fill="{p['accent']}" font-size="12" font-style="italic" font-weight="600" font-family="system-ui, -apple-system, sans-serif">
    {sci}
  </text>
  
  <text x="32" y="454" fill="#94a3b8" font-size="10" font-family="system-ui, -apple-system, sans-serif">
    Family: {family}
  </text>
</svg>
"""
    return svg.strip()

def main():
    with open(SPECIES_FILE, "r", encoding="utf-8") as f:
        species_list = json.load(f)

    count = 0
    for tree in species_list:
        tid = tree.get("id", "")
        if not tid:
            continue
        svg_content = generate_svg(tree)
        filepath = os.path.join(OUTPUT_DIR, f"{tid}.svg")
        with open(filepath, "w", encoding="utf-8") as out_f:
            out_f.write(svg_content)
        count += 1

    print(f"[OK] Successfully generated {count} tree images in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
