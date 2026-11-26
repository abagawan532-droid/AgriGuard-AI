import json
import os
import re

tree_raw_names = [
    "Neem", "Banyan", "Peepal", "Mango", "Coconut", "Arecanut", "Teak", "Sandalwood", "Eucalyptus", "Rosewood",
    "Tamarind", "Jackfruit", "Jamun", "Arjuna", "Ashoka", "Gulmohar", "Pongamia", "Rain tree", "Mahogany", "Bamboo",
    "Acacia", "Subabul", "Drumstick", "Curry leaf", "Amla", "Guava", "Papaya", "Banana", "Pomegranate", "Sapota",
    "Custard apple", "Lemon", "Orange", "Sweet lime", "Grapes", "Fig", "Mulberry", "Cashew", "Avocado", "Ber",
    "Bael", "Wood apple", "Indian almond", "Bottlebrush", "Plumeria", "Hibiscus", "Jasmine", "Rose", "Marigold", "Chrysanthemum",
    "Tuberose", "Crossandra", "Bougainvillea", "Periwinkle", "Ixora", "Oleander", "Gardenia", "Night jasmine", "Butterfly pea", "Aparajita",
    "Tulsi", "Aloe vera", "Ashwagandha", "Brahmi", "Shatavari", "Lemongrass", "Vetiver", "Mint", "Coriander", "Fenugreek",
    "Ginger", "Turmeric", "Black pepper", "Cardamom", "Clove", "Cinnamon", "Curry tree", "Betel leaf", "Betel nut palm", "Coffee plant",
    "Tea plant", "Cocoa", "Rubber tree", "Cotton", "Sugarcane", "Tobacco", "Ragi", "Jowar", "Maize", "Rice",
    "Wheat", "Bajra", "Sunflower", "Groundnut", "Sesame", "Castor", "Mustard", "Safflower", "Soybean", "Red gram",
    "Green gram", "Black gram", "Bengal gram", "Horse gram", "Cowpea", "Field bean", "Pea", "Tomato", "Brinjal", "Chilli",
    "Capsicum", "Potato", "Sweet potato", "Onion", "Garlic", "Cabbage", "Cauliflower", "Carrot", "Radish", "Beetroot",
    "Okra", "Bottle gourd", "Bitter gourd", "Ridge gourd", "Snake gourd", "Ash gourd", "Pumpkin", "Cucumber", "Ivy gourd", "Cluster bean",
    "French bean", "Spinach", "Amaranthus", "Drumstick tree", "Colocasia", "Tapioca", "Elephant foot yam", "Taro", "Knol-khol", "Celery",
    "Lettuce", "Broccoli", "Kale", "Swiss chard", "Parsley", "Dill", "Fennel", "Cumin", "Ajwain", "Mustard plant",
    "Lotus", "Water lily", "Money plant", "Snake plant", "Areca palm", "Date palm", "Foxtail palm", "Fan palm", "Traveller's palm", "Rubber plant",
    "Croton", "Dieffenbachia", "Dracaena", "Peace lily", "Spider plant", "Fern", "Boston fern", "Anthurium", "Orchid", "Gerbera",
    "Dahlia", "Petunia", "Zinnia", "Cosmos", "Sunflower (Garden)", "Calendula", "Balsam", "Portulaca", "Morning glory", "Rangoon creeper",
    "Passion flower", "Tecoma", "Golden shower tree", "Indian cork tree", "Flame of the forest", "Indian beech", "Kadamba", "Honge", "Bilva", "Nagalinga tree",
    "Baobab", "Tamarind tree", "Indian gooseberry", "Malabar nut", "Adulsa", "Giloy", "Insulin plant", "Patharchatta", "Shankhpushpi", "Harsingar"
]

# Curated scientific data and categorization map
SPECIES_META = {
    "neem": ("Azadirachta indica", "Meliaceae", "timber", "Native to Indian subcontinent; widely planted in tropics", "Tropical & semi-arid", "Well-drained loam", "Low", 22.5, 1200, "🌿"),
    "banyan": ("Ficus benghalensis", "Moraceae", "sacred", "Indian subcontinent; iconic aerial prop roots", "Warm tropical", "Adaptable, well-drained", "Moderate", 48.0, 2500, "🌳"),
    "peepal": ("Ficus religiosa", "Moraceae", "sacred", "Indian subcontinent & Indochina; revered sacred tree", "Tropical to subtropical", "Porous, rocky to loam", "Low", 35.0, 2000, "🍃"),
    "mango": ("Mangifera indica", "Anacardiaceae", "fruit", "South Asia; world's leading tropical fruit tree", "Tropical & subtropical", "Deep fertile loam", "Moderate", 26.5, 6500, "🥭"),
    "coconut": ("Cocos nucifera", "Arecaceae", "plantation", "Humid tropical coastal and inland plains", "Warm humid tropical", "Sandy coastal, red loam", "High", 20.0, 3500, "🥥"),
    "arecanut": ("Areca catechu", "Arecaceae", "plantation", "Tropical Asia; prominent betel nut palm", "Humid tropical", "Laterite & clay loam", "High", 18.0, 4200, "🌰"),
    "teak": ("Tectona grandis", "Lamiaceae", "timber", "South & Southeast Asia; gold-standard luxury timber", "Tropical with dry seasons", "Deep alluvial rich loam", "Moderate", 31.0, 4500, "🪵"),
    "sandalwood": ("Santalum album", "Santalaceae", "timber", "Southern India; high-value aromatic heartwood", "Dry deciduous plains", "Red loam, well-drained", "Low", 18.0, 12000, "🪵"),
    "eucalyptus": ("Eucalyptus globulus", "Myrtaceae", "timber", "Australia; fast-growing pulpwood & pole plantation", "Temperate to semi-arid", "Well-drained soils", "Moderate", 24.0, 1800, "🌲"),
    "rosewood": ("Dalbergia sissoo", "Fabaceae", "timber", "Foothills of Himalayas & North/Central India", "Subtropical & tropical", "Alluvial sandy loam", "Moderate", 27.0, 4000, "🪵"),
    "tamarind": ("Tamarindus indica", "Fabaceae", "fruit", "Tropical Africa & South Asia; tart fruit and heavy wood", "Dry tropical", "Deep alluvial & red loam", "Low", 28.0, 3200, "🫘"),
    "jackfruit": ("Artocarpus heterophyllus", "Moraceae", "fruit", "Western Ghats of India; largest tree-borne fruit", "Warm humid tropical", "Deep rich laterite", "Moderate", 25.0, 5200, "🍈"),
    "jamun": ("Syzygium cumini", "Myrtaceae", "fruit", "Indian subcontinent; nutritious purple berries & timber", "Tropical & subtropical", "Alluvial & loamy soil", "Moderate", 24.0, 3400, "🫐"),
    "arjuna": ("Terminalia arjuna", "Combretaceae", "medicinal", "Riverbanks of central & southern India; cardio-tonic bark", "Subtropical & tropical", "Fertile alluvial river loam", "High", 33.0, 2600, "🌳"),
    "ashoka": ("Saraca asoca", "Fabaceae", "sacred", "Western Ghats & central India; revered sacred flowering tree", "Humid tropical rain-forest", "Moist well-drained loam", "Moderate", 20.0, 2800, "🌸"),
    "gulmohar": ("Delonix regia", "Fabaceae", "ornamental", "Madagascar; vibrant scarlet canopy avenue tree", "Warm tropical", "Well-drained porous soil", "Moderate", 21.0, 1500, "🌺"),
    "pongamia": ("Millettia pinnata", "Fabaceae", "timber", "Indian subcontinent; biofuel seed oil & nitrogen fixing", "Tropical & coastal", "Tolerates saline & rocky", "Low", 23.0, 2200, "🌳"),
    "rain-tree": ("Albizia saman", "Fabaceae", "ornamental", "Tropical Americas; gigantic umbrella canopy", "Warm humid tropical", "Deep rich moist loam", "Moderate", 44.0, 2000, "🌳"),
    "mahogany": ("Swietenia macrophylla", "Meliaceae", "timber", "Central & South America; premier furniture wood", "Humid tropical", "Deep rich alluvial loam", "Moderate", 28.0, 3800, "🪵"),
    "bamboo": ("Dendrocalamus strictus", "Poaceae", "timber", "Tropical Asia; fast-growing giant woody grass", "Tropical to subtropical", "Well-drained sandy loam", "Moderate", 30.0, 800, "🎋"),
    "acacia": ("Acacia nilotica (Babul)", "Fabaceae", "timber", "Arid and semi-arid plains of India and Africa", "Dry arid & desert", "Black cotton & sandy loam", "Low", 19.0, 1900, "🌳"),
    "subabul": ("Leucaena leucocephala", "Fabaceae", "timber", "Central America; rapid biomass and green forage", "Warm tropical", "Neutral to alkaline loam", "Low", 22.0, 1200, "🌿"),
    "drumstick": ("Moringa oleifera", "Moringaceae", "medicinal", "Sub-Himalayan tracts; superfood miracle tree", "Tropical & subtropical", "Porous sandy loam", "Low", 16.0, 2500, "🎋"),
    "curry-leaf": ("Murraya koenigii", "Rutaceae", "medicinal", "Indian subcontinent; aromatic culinary tree", "Tropical to warm temperate", "Well-drained red loam", "Moderate", 12.0, 1800, "🍃"),
    "amla": ("Phyllanthus emblica", "Phyllanthaceae", "medicinal", "Indian forests; rich Vitamin-C Ayurvedic fruit", "Tropical & dry subtropical", "Light to heavy alkaline loam", "Low", 20.0, 3200, "🍏"),
    "guava": ("Psidium guajava", "Myrtaceae", "fruit", "Tropical America; hardy orchard fruit tree", "Tropical & subtropical", "Wide range, loam to clay", "Moderate", 18.0, 2800, "🍏"),
    "papaya": ("Carica papaya", "Caricaceae", "fruit", "Tropical America; fast-yielding soft-wooded fruit tree", "Humid tropical", "Porous sandy loam", "Moderate", 14.0, 2200, "🍈"),
    "banana": ("Musa acuminata", "Musaceae", "fruit", "Indomalaya; world's largest herbaceous perennial", "Humid tropical", "Deep rich alluvial loam", "High", 15.0, 1800, "🍌"),
    "pomegranate": ("Punica granatum", "Lythraceae", "fruit", "Mediterranean to Central Asia; antioxidant-rich fruit", "Semi-arid tropical", "Light deep loam", "Moderate", 16.0, 4800, "🍎"),
    "sapota": ("Manilkara zapota", "Sapotaceae", "fruit", "Central America; hardy sweet brown fruit tree", "Tropical coastal & plains", "Alluvial, laterite, sandy loam", "Moderate", 22.0, 2600, "🥔"),
    "custard-apple": ("Annona squamosa", "Annonaceae", "fruit", "Tropical Americas; creamy dessert fruit tree", "Dry tropical & rocky", "Shallow gravelly loam", "Low", 15.0, 3500, "🍏"),
    "lemon": ("Citrus limon", "Rutaceae", "fruit", "South Asia; evergreen sour citrus tree", "Warm subtropical", "Light well-drained loam", "Moderate", 14.0, 3000, "🍋"),
    "orange": ("Citrus sinensis", "Rutaceae", "fruit", "Southern China & India; sweet citrus orchard tree", "Subtropical & tropical", "Deep fertile sandy loam", "Moderate", 16.0, 3600, "🍊"),
    "sweet-lime": ("Citrus limetta", "Rutaceae", "fruit", "South Asia; refreshing mild sweet citrus", "Tropical & subtropical", "Deep well-drained loam", "Moderate", 15.0, 3400, "🍋"),
    "grapes": ("Vitis vinifera", "Vitaceae", "fruit", "Mediterranean & Central Asia; commercial woody vine", "Subtropical dry summer", "Sandy loam to medium black", "Moderate", 12.0, 5500, "🍇"),
    "fig": ("Ficus carica", "Moraceae", "fruit", "Mediterranean & Middle East; sweet nutritious fruit tree", "Warm dry subtropical", "Well-drained light loam", "Low", 16.0, 4200, "🫐"),
    "mulberry": ("Morus alba", "Moraceae", "fruit", "East Asia; sweet berries & sericulture food tree", "Temperate to tropical", "Deep fertile loamy soil", "Moderate", 19.0, 2400, "🫐"),
    "cashew": ("Anacardium occidentale", "Anacardiaceae", "fruit", "Northeastern Brazil; tropical nut & apple tree", "Warm tropical coastal", "Porous red laterite", "Low", 20.0, 3800, "🥜"),
    "avocado": ("Persea americana", "Lauraceae", "fruit", "Central America; rich buttery evergreen tree", "Warm subtropical", "Well-aerated volcanic loam", "Moderate", 22.0, 6000, "🥑"),
    "ber": ("Ziziphus mauritiana", "Rhamnaceae", "fruit", "Arid tropical regions; drought-hardy jujube tree", "Hot arid & semi-arid", "Gravelly sandy soils", "Low", 17.0, 1800, "🫒"),
    "bael": ("Aegle marmelos", "Rutaceae", "sacred", "Indian subcontinent; sacred Ayurvedic aromatic fruit tree", "Subtropical & dry tropical", "Porous clay loam", "Low", 20.0, 2600, "🍏"),
    "wood-apple": ("Limonia acidissima", "Rutaceae", "fruit", "Indian dry forests; hard-shelled sour aromatic fruit", "Dry deciduous tropical", "Rocky dry soils", "Low", 18.0, 2100, "🍏"),
    "indian-almond": ("Terminalia catappa", "Combretaceae", "ornamental", "Coastal tropical Asia & Pacific; umbrella coastal canopy", "Warm tropical coastal", "Sandy coastal soils", "Moderate", 26.0, 2400, "🍃"),
    "bottlebrush": ("Callistemon citrinus", "Myrtaceae", "ornamental", "Australia; vibrant crimson bottle-brush flowers", "Temperate to tropical", "Moist loamy soil", "Moderate", 14.0, 1400, "🌺"),
    "plumeria": ("Plumeria rubra (Champa)", "Apocynaceae", "ornamental", "Tropical Americas; fragrant temple floral tree", "Warm tropical", "Well-drained dryish loam", "Low", 15.0, 1600, "🌸"),
    "golden-shower-tree": ("Cassia fistula (Amaltas)", "Fabaceae", "ornamental", "Indian subcontinent; cascading golden yellow blossoms", "Tropical & subtropical", "Well-drained light loam", "Low", 22.0, 2200, "🌼"),
    "indian-cork-tree": ("Millingtonia hortensis", "Bignoniaceae", "ornamental", "Southeast Asia; tall slender tree with fragrant white blooms", "Tropical to subtropical", "Deep moist loam", "Moderate", 24.0, 2100, "🌸"),
    "flame-of-the-forest": ("Butea monosperma (Palash)", "Fabaceae", "sacred", "South Asia; brilliant orange-red springtime bloom", "Dry deciduous plains", "Heavy saline clay to rocky", "Low", 23.0, 1900, "🔥"),
    "kadamba": ("Neolamarckia cadamba", "Rubiaceae", "sacred", "South Asia; sacred flowering tree with globe flowers", "Humid tropical rain-forest", "Deep rich moist alluvial", "High", 32.0, 2800, "🟠"),
    "nagalinga-tree": ("Couroupita guianensis", "Lecythidaceae", "sacred", "Tropical rainforests; cannonball tree with sacred blooms", "Warm humid tropical", "Deep rich moist loam", "High", 29.0, 3000, "🌺"),
    "baobab": ("Adansonia digitata", "Malvaceae", "sacred", "African savannah; colossal bottle-trunk water-storing tree", "Hot arid & dry tropical", "Sandy rocky desert loam", "Low", 42.0, 3000, "🏺")
}

species_list = []
used_ids = set()

for i, raw_name in enumerate(tree_raw_names, start=1):
    slug = re.sub(r"[^a-z0-9]+", "-", raw_name.lower()).strip("-")
    if slug in used_ids:
        slug = f"{slug}-{i}"
    used_ids.add(slug)
    
    # Check if we have specialized metadata
    lookup_key = slug.split("-")[0]
    meta = SPECIES_META.get(slug, SPECIES_META.get(lookup_key, None))
    
    if meta:
        scientific, family, category, dist, climate, soil, water, carbon, price, icon = meta
    else:
        # Determine logical category and parameters
        name_lower = raw_name.lower()
        if any(w in name_lower for w in ["palm", "fern", "bamboo"]):
            category = "palm"
            icon = "🌴"
            carbon = 16.0
            price = 2200
        elif any(w in name_lower for w in ["rose", "jasmine", "marigold", "chrysanthemum", "tuberose", "crossandra", "gerbera", "dahlia", "petunia", "zinnia", "cosmos", "calendula", "balsam", "lotus", "lily", "orchid", "anthurium", "hibiscus", "bougainvillea", "plumeria", "oleander"]):
            category = "ornamental"
            icon = "🌸"
            carbon = 8.0
            price = 1500
        elif any(w in name_lower for w in ["tulsi", "aloe", "ashwagandha", "brahmi", "shatavari", "neem", "giloy", "adulsa", "malabar", "shankhpushpi", "harsingar", "patharchatta", "insulin", "mint", "curry", "ginger", "turmeric", "pepper", "clove", "cinnamon"]):
            category = "medicinal"
            icon = "🌿"
            carbon = 12.0
            price = 2800
        elif any(w in name_lower for w in ["mango", "apple", "banana", "guava", "papaya", "sapota", "orange", "lemon", "lime", "fig", "mulberry", "cashew", "avocado", "ber", "pomegranate"]):
            category = "fruit"
            icon = "🥭"
            carbon = 20.0
            price = 3500
        elif any(w in name_lower for w in ["teak", "sandalwood", "rosewood", "mahogany", "eucalyptus", "acacia", "oak", "timber", "pine", "beech", "cork"]):
            category = "timber"
            icon = "🪵"
            carbon = 28.0
            price = 4500
        elif any(w in name_lower for w in ["peepal", "banyan", "ashoka", "kadamba", "nagalinga", "bilva", "flame"]):
            category = "sacred"
            icon = "🌳"
            carbon = 32.0
            price = 2400
        else:
            category = "plantation"
            icon = "🌱"
            carbon = 14.0
            price = 1800
            
        scientific = f"{raw_name} sp."
        family = "Botanical Flora"
        dist = "Native and widely cultivated across Indian and tropical eco-zones"
        climate = "Tropical to subtropical climate with moderate rainfall"
        soil = "Fertile well-drained loam or alluvial soil"
        water = "Moderate"
        
    species_list.append({
        "id": slug,
        "num": i,
        "common_name": raw_name,
        "scientific_name": scientific,
        "family": family,
        "category": category,
        "distribution": dist,
        "climate": climate,
        "soil": soil,
        "water": water,
        "sunlight": "Full sun to partial canopy",
        "uses": f"Ecological conservation, agroforestry, medicinal, timber, and environmental shade benefits of {raw_name}.",
        "growth_maturity": "Variable rotation cycle; reaches prime vegetative and reproductive maturity with proper management.",
        "age_note": "Accurate age determination requires planting documentation, trunk core sampling, or ring counts.",
        "value_note": f"Valuation for {raw_name} depends on local demand, timber volume, harvest grade, and market trends.",
        "default_unit_price": price,
        "carbon_rate_kg_yr": carbon,
        "icon_emoji": icon
    })

# Write updated tree_species.json
os.makedirs("data", exist_ok=True)
with open("data/tree_species.json", "w", encoding="utf-8") as f:
    json.dump(species_list, f, ensure_ascii=False, indent=2)

# Write updated tree_labels.json
tree_ids = [t["id"] for t in species_list]
with open("data/tree_labels.json", "w", encoding="utf-8") as f:
    json.dump(tree_ids, f, ensure_ascii=False, indent=2)

print(f"Generated data/tree_species.json and data/tree_labels.json with {len(species_list)} tree species!")
