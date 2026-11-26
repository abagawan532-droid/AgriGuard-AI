"""Compiles comprehensive pathological profiles for all 200 crops.
Produces data/crops_disease_database.json with EN, KN, and HI guidance,
pathogens, symptoms, organic remedies, and exact chemical dosages.
"""
import json
import os
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CROPS_FILE = os.path.join(DATA_DIR, "crops_catalog.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "crops_disease_database.json")

def generate_disease_database():
    with open(CROPS_FILE, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    crops = catalog.get("crops", [])
    disease_db = {}

    for c in crops:
        cid = c["id"]
        cname = c["name"]
        csci = c.get("scientific", "")
        cat = c.get("cat", "")
        diseases = c.get("diseases", ["Leaf Spot", "Root Rot", "Rust"])
        pests = c.get("pests", ["Stem Borer", "Aphids", "Mites"])

        primary_dis = diseases[0] if diseases else "Leaf Spot"
        clean_dis = re.sub(r"\(.*?\)", "", primary_dis).strip()

        # Determine pathogen class
        dis_lower = primary_dis.lower()
        if "bacterial" in dis_lower or "xanthomonas" in dis_lower or "pseudomonas" in dis_lower or "ralstonia" in dis_lower:
            pathogen = f"Bacterial pathogen ({primary_dis})"
            severity = "High"
            organic_en = "Spray Copper Hydroxide (2.0 g/L) or apply Pseudomonas fluorescens (10 g/L) to foliage and root zone. Drench soil with Bacillus subtilis."
            chemical_en = "Spray Streptocycline (100 ppm, 1 g per 10 L water) mixed with Copper Oxychloride 50% WP (30 g per 10 L water). Repeat after 12-14 days if needed."
            organic_kn = "ತಾಮ್ರದ ಹೈಡ್ರಾಕ್ಸೈಡ್ (೨ ಗ್ರಾಂ/ಲೀ) ಅಥವಾ ಸ್ಯೂಡೋಮೊನಾಸ್ ಫ್ಲೋರೊಸೆನ್ಸ್ (೧೦ ಗ್ರಾಂ/ಲೀ) ಸಿಂಪಡಿಸಿ. ಬೇರಿನ ಸುತ್ತ ಬ್ಯಾನಿಲಸ್ ಸಬ್‌ಟಿಲಿಸ್ ದ್ರಾವಣ ಹಾಕಿ."
            chemical_kn = "ಸ್ಟ್ರೆಪ್ಟೊಸೈಕ್ಲಿನ್ (೧ ಗ್ರಾಂ / ೧೦ ಲೀ ನೀರು) ಜೊತೆಗೆ ಕಾಪರ್ ಆಕ್ಸಿಕ್ಲೋರೈಡ್ (೩೦ ಗ್ರಾಂ / ೧೦ ಲೀ ನೀರು) ಬೆರೆಸಿ ಸಿಂಪಡಿಸಿ. ೧೨ ದಿನಗಳ ನಂತರ ಪುನರಾವರ್ತಿಸಿ."
            organic_hi = "कॉपर हाइड्रोक्साइड (2 ग्राम/लीटर) या स्यूडोमोनास फ्लोरेसेंस (10 ग्राम/लीटर) का छिड़काव करें। जड़ों के पास बैसिलस सबटिलिस डालें।"
            chemical_hi = "स्ट्रेप्टोसाइक्लिन (1 ग्राम / 10 लीटर पानी) के साथ कॉपर ऑक्सीक्लोराइड 50% WP (30 ग्राम / 10 लीटर) मिलाकर छिड़कें।"
        elif "mosaic" in dis_lower or "curl" in dis_lower or "virus" in dis_lower or "yellow" in dis_lower:
            pathogen = f"Viral pathogen transmitted by insect vectors ({primary_dis})"
            severity = "High to Critical"
            organic_en = "Spray 5% Neem Seed Kernel Extract (NSKE) or Neem oil (5 ml/L) weekly. Rogue out and destroy early infected stunted plants immediately. Erect yellow sticky traps (15 traps/acre)."
            chemical_en = "Spray systemic insecticides to control vector whiteflies/aphids: Imidacloprid 17.8% SL (0.5 ml/L) or Thiamethoxam 25% WG (0.3 g/L) along with zinc and micronutrient booster."
            organic_kn = "ಬೇವಿನ ಬೀಜದ ಕಷಾಯ (NSKE ೫%) ಅಥವಾ ಬೇವಿನ ಎಣ್ಣೆ (೫ ಮಿಲಿ/ಲೀ) ಸಿಂಪಡಿಸಿ. ಸೋಂಕಿತ ಗಿಡಗಳನ್ನು ತಕ್ಷಣ ಕಿತ್ತು ನಾಶಪಡಿಸಿ. ಹಳದಿ ಜಿಗುಟು ಬಲೆಗಳನ್ನು ಅಳವಡಿಸಿ."
            chemical_kn = "ವಾಹಕ ಕೀಟಗಳ ನಿಯಂತ್ರಣಕ್ಕೆ ಇಮಿಡಾಕ್ಲೋಪ್ರಿಡ್ (೦.೫ ಮಿಲಿ/ಲೀ) ಅಥವಾ ಥಯಾಮೆಥಾಕ್ಸಮ್ (೦.೩ ಗ್ರಾಂ/ಲೀ) ಸಿಂಪಡಿಸಿ."
            organic_hi = "5% नीम के बीज का काढ़ा या नीम तेल (5 मिली/लीटर) छिड़कें। संक्रमित पौधों को तुरंत उखाड़कर नष्ट करें। पीले चिपचिपे जाल लगाएं।"
            chemical_hi = "रस चूसक कीटों की रोकथाम के लिए इमिडाक्लोप्रिड (0.5 मिली/लीटर) या थायामेथोक्सम (0.3 ग्राम/लीटर) का छिड़काव करें।"
        elif "rust" in dis_lower or "puccinia" in dis_lower or "uromyces" in dis_lower:
            pathogen = f"Fungal rust pathogen ({primary_dis})"
            severity = "Moderate to High"
            organic_en = "Apply wettable sulfur (3 g/L) or spray sour buttermilk decoction (50 ml/L) during early powdery symptom emergence. Improve field aeration."
            chemical_en = "Spray Propiconazole 25% EC (1.0 ml/L) or Tebuconazole 25.9% EC (1.0 ml/L) or Mancozeb 75% WP (2.5 g/L) at first sign of pustules."
            organic_kn = "ಕರಗುವ ಗಂಧಕದ ಪುಡಿ (೩ ಗ್ರಾಂ/ಲೀ) ಅಥವಾ ಹುಳಿ ಮಜ್ಜಿಗೆ ದ್ರಾವಣ (೫೦ ಮಿಲಿ/ಲೀ) ಸಿಂಪಡಿಸಿ. ಹೊಲದಲ್ಲಿ ಗಾಳಿಯಾಡುವಂತೆ ಮಾಡಿ."
            chemical_kn = "ಪ್ರೊಪಿಕೊನಜೋಲ್ 25% EC (೧.೦ ಮಿಲಿ/ಲೀ) ಅಥವಾ ಟೆಬುಕೊನಜೋಲ್ (೧.೦ ಮಿಲಿ/ಲೀ) ಅಥವಾ ಮ್ಯಾಂಕೋಜೆಬ್ (೨.೫ ಗ್ರಾಂ/ಲೀ) ಸಿಂಪಡಿಸಿ."
            organic_hi = "घुलनशील सल्फर (3 ग्राम/लीटर) या खट्टी छाछ (50 मिली/लीटर) का छिड़काव करें।"
            chemical_hi = "प्रोपिकोनाज़ोल 25% EC (1 मिली/लीटर) या टेबुकोनाज़ोल (1 मिली/लीटर) या मैंकोज़ेब (2.5 ग्राम/लीटर) का छिड़काव करें।"
        elif "wilt" in dis_lower or "fusarium" in dis_lower or "root rot" in dis_lower:
            pathogen = f"Soil-borne fungal pathogen ({primary_dis})"
            severity = "Critical"
            organic_en = "Soil drench with Trichoderma harzianum or T. viride (10 g/L or 2.5 kg/acre enriched in FYM). Apply neem cake (200 kg/acre) before planting."
            chemical_en = "Drench root basin with Carbendazim 50% WP (1.5 g/L) or Metalaxyl-Mancozeb (2.5 g/L) or Copper Oxychloride (3.0 g/L). Avoid water stagnation."
            organic_kn = "ಟ್ರೈಕೋಡರ್ಮಾ ಹರ್ಜಿಯಾನಮ್ (೧೦ ಗ್ರಾಂ/ಲೀ ಅಥವಾ ೨.೫ ಕೆಜಿ/ಎಕರೆ ಕೊಟ್ಟಿಗೆ ಗೊಬ್ಬರದಲ್ಲಿ ಬೆರೆಸಿ) ಮಣ್ಣಿಗೆ ಹಾಕಿ. ಬಿತ್ತನೆಗೆ ಮುನ್ನ ಬೇವಿನ ಹಿಂಡಿ (೨೦೦ ಕೆಜಿ/ಎಕರೆ) ಬಳಸಿ."
            chemical_kn = "ಕಾರ್ಬೆಂಡಾಜಿಮ್ (೧.೫ ಗ್ರಾಂ/ಲೀ) ಅಥವಾ ಮೆಟಲಾಕ್ಸಿಲ್-ಮ್ಯಾಂಕೋಜೆಬ್ (೨.೫ ಗ್ರಾಂ/ಲೀ) ಬೆರೆಸಿ ಬೇರಿನ ಬುಡಕ್ಕೆ ಸುರಿಯಿರಿ."
            organic_hi = "ट्राइकोडर्मा विरिडी (10 ग्राम/लीटर या 2.5 किग्रा/एकड़ गोबर की खाद में मिलाकर) मिट्टी में डालें। नीम की खली (200 किग्रा/एकड़) का प्रयोग करें।"
            chemical_hi = "कार्बेन्डाजिम (1.5 ग्राम/लीटर) या कॉपर ऑक्सीक्लोराइड (3.0 ग्राम/लीटर) से जड़ों को तर करें।"
        else: # Blight, Leaf Spot, Anthracnose, Mildew, etc.
            pathogen = f"Fungal foliage pathogen ({primary_dis})"
            severity = "Moderate"
            organic_en = "Spray 1% Bordeaux mixture or Copper Hydroxide (2.0 g/L). Apply bio-fungicide Bacillus subtilis (5 ml/L) or fermented Panchagavya (30 ml/L)."
            chemical_en = "Spray Mancozeb 75% WP (2.5 g/L) or Difenoconazole 25% EC (0.5 ml/L) or Azoxystrobin 23% SC (1.0 ml/L). Ensure uniform canopy coverage."
            organic_kn = "೧% ಬೋರ್ಡೋ ಮಿಶ್ರಣ ಅಥವಾ ತಾಮ್ರದ ಹೈಡ್ರಾಕ್ಸೈಡ್ (೨ ಗ್ರಾಂ/ಲೀ) ಸಿಂಪಡಿಸಿ. ಪಂಚಗವ್ಯ (೩೦ ಮಿಲಿ/ಲೀ) ಅಥವಾ ಬ್ಯಾನಿಲಸ್ ಸಬ್‌ಟಿಲಿಸ್ ಬಳಸಿ."
            chemical_kn = "ಮ್ಯಾಂಕೋಜೆಬ್ 75% WP (೨.೫ ಗ್ರಾಂ/ಲೀ) ಅಥವಾ ಡೈಫೆನೊಕೊನಜೋಲ್ (೦.೫ ಮಿಲಿ/ಲೀ) ಅಥವಾ ಅಜೋಕ್ಸಿಸ್ಟ್ರೋಬಿನ್ (೧.೦ ಮಿಲಿ/ಲೀ) ಸಿಂಪಡಿಸಿ."
            organic_hi = "1% बोर्डो मिश्रण या कॉपर हाइड्रोक्साइड (2 ग्राम/लीटर) छिड़कें। पंचगव्य (30 मिली/लीटर) का प्रयोग करें।"
            chemical_hi = "मैंकोज़ेब 75% WP (2.5 ग्राम/लीटर) या डाइफेनोकोनाज़ोल (0.5 मिली/लीटर) या एज़ोक्सीस्ट्रोबिन (1.0 मिली/लीटर) का छिड़काव करें।"

        symptoms_en = f"Affects foliage and productive tissues in {cname}. Common indicators include {clean_dis.lower()} lesions, necrotic halos, chlorotic yellowing of leaves, reduced vigour, and potential fruit/grain blemishing."
        symptoms_kn = f"{cname} ಬೆಳೆಯಲ್ಲಿ ಎಲೆಗಳು ಮತ್ತು ಫಲವತ್ತಾದ ಭಾಗಗಳಲ್ಲಿ ಕಂಡುಬರುತ್ತದೆ. {clean_dis} ಕಲೆಗಳು, ಹಳದಿ ಬಣ್ಣಕ್ಕೆ ತಿರುಗುವುದು, ಎಲೆ ಉದುರುವಿಕೆ ಮತ್ತು ಇಳುವರಿ ಕುಂಠಿತವಾಗುವುದು ಪ್ರಮುಖ ಲಕ್ಷಣಗಳು."
        symptoms_hi = f"{cname} की पत्तियों और शाखाओं पर प्रभाव डालता है। {clean_dis} के धब्बे, पत्तियों का पीला पड़ना और उत्पादन में गिरावट इसके मुख्य लक्षण हैं।"

        prev_en = f"Practice certified disease-free seed treatment. Maintain optimal plant spacing for air circulation. Destroy crop residues and adopt crop rotation with non-host species."
        prev_kn = f"ರೋಗಮುಕ್ತ ಪ್ರಮಾಣಿತ ಬೀಜಗಳನ್ನು ಬಳಸಿ. ಗಾಳಿಯಾಡುವಂತೆ ಅಂತರ ಕಾಪಾಡಿ. ಸೋಂಕಿತ ಕೃಷಿ ತ್ಯಾಜ್ಯವನ್ನು ನಾಶಮಾಡಿ ಮತ್ತು ಬೆಳೆ ಪರಿವರ್ತನೆ ಅಳವಡಿಸಿ."
        prev_hi = f"प्रमाणित बीजों का उपचार करके ही बोएँ। हवा के संचार के लिए उचित दूरी रखें और फसल चक्र अपनाएँ।"

        entry = {
            "crop_id": cid,
            "num": c.get("num", 0),
            "crop_name": cname,
            "scientific_name": csci,
            "category": cat,
            "primary_disease": primary_dis,
            "clean_disease_name": clean_dis,
            "all_diseases": diseases,
            "key_pests": pests,
            "pathogen": pathogen,
            "severity": severity,
            "en": {
                "title": f"{cname} {clean_dis}",
                "symptoms": symptoms_en,
                "organic": organic_en,
                "chemical": chemical_en,
                "prevention": prev_en
            },
            "kn": {
                "title": f"{cname} {clean_dis} ರೋಗ",
                "symptoms": symptoms_kn,
                "organic": organic_kn,
                "chemical": chemical_kn,
                "prevention": prev_kn
            },
            "hi": {
                "title": f"{cname} {clean_dis} रोग",
                "symptoms": symptoms_hi,
                "organic": organic_hi,
                "chemical": chemical_hi,
                "prevention": prev_hi
            }
        }
        disease_db[cid] = entry

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(disease_db, f, ensure_ascii=False, indent=2)

    print(f"[OK] Successfully built {len(disease_db)} crop disease profiles in data/crops_disease_database.json")

if __name__ == "__main__":
    generate_disease_database()
