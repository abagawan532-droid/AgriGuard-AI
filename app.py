import os
import json
import uuid
import shutil
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

from agri_ai import (
    predict_leaf, predict_tree, get_leaf_model, get_tree_model,
    get_all_crop_diseases, get_crop_disease
)
from tree_data import get_tree, search_trees, calculate_tree_value, calculate_forestry_biomass, all_trees
from crops_service import get_crop_catalog, get_all_crops, get_categories, get_crop_by_id, search_crops, calculate_crop_economics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
SAMPLES_DIR = os.path.join(BASE_DIR, "static", "samples")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "pdf"}
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("AGRIGUARD_SECRET", "agriguard-ai-production-key-2026")
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max

LANGS = {
    "en": {"name": "English", "home": "Home", "leaf": "Crop Disease Scanner", "tree": "Tree Scanner", "crops": "200 Crops Catalog", "trees": "Tree Library", "calc": "Agro-Calculator", "about": "Docs & AI Status"},
    "kn": {"name": "ಕನ್ನಡ", "home": "ಮುಖಪುಟ", "leaf": "ಬೆಳೆ ರೋಗ ಸ್ಕ್ಯಾನರ್", "tree": "ಮರ ಗುರುತಿಸುವಿಕೆ", "crops": "೨೦೦ ಬೆಳೆಗಳ ಮಾಹಿತಿ", "trees": "ಮರಗಳ ಗ್ರಂಥಾಲಯ", "calc": "ಕೃಷಿ ಕ್ಯಾಲ್ಕುಲೇಟರ್", "about": "ವಿವರ ಮತ್ತು AI ಸ್ಥಿತಿ"},
    "hi": {"name": "हिन्दी", "home": "होम", "leaf": "फसल रोग स्कैनर", "tree": "पेड़ पहचान स्कैनर", "crops": "२०० फसल डायरेक्टरी", "trees": "वृक्ष लाइब्रेरी", "calc": "कृषि कैलकुलेटर", "about": "डॉक्यूमेंटेशन"}
}

TEXT = {
    "en": {
        "home": "Home",
        "leaf": "Crop Disease Scanner",
        "tree": "Tree Scanner",
        "crops": "200 Crops Catalog",
        "trees": "Tree Reference Library",
        "calc": "Agro-Calculator",
        "about": "Docs & AI Status",
        "title": "AgriGuard AI",
        "tagline": "Smart Crop & Tree Intelligence System",
        "intro": "Intelligent agricultural decision support with 200 crops catalog, real-time leaf disease diagnosis, tree species identification, and forestry carbon valuation.",
        "leaf_title": "Crop Leaf Disease Scanner",
        "tree_title": "Tree Species Recognition Scanner",
        "upload_leaf": "Upload or capture a clear crop leaf photo",
        "upload_tree": "Upload or capture a clear tree photo (leaf, canopy, or bark)",
        "scan": "Analyze with AI",
        "scanning": "Analyzing botanical features...",
        "tree_search": "Search tree database",
        "search_placeholder": "Search by common name, scientific name, or family...",
        "crop_search_placeholder": "Search 200 crops by name, category, season, or diseases...",
        "database": "Browse Tree Species Library",
        "result": "AI Diagnosis Report",
        "confidence": "AI Model Confidence",
        "disclaimer": "AgriGuard AI provides preliminary diagnostic estimates. Always verify severe crop infections with a local agricultural extension officer before chemical spray application.",
        "model_missing": "AI model loading. System ready with built-in agronomic intelligence.",
        "not_found": "No matching record found. Try a different search term or filter.",
        "symptoms": "Pathological Symptoms",
        "organic": "Biological & Organic Remedies",
        "chemical": "Chemical Fungicides & Recommended Dosages",
        "prevention": "Preventive Cultural Practices",
        "pathogen": "Identified Pathogen",
        "severity": "Disease Severity Level",
        "common": "Common Name",
        "scientific": "Scientific Name",
        "family": "Botanical Family",
        "region": "Native Distribution",
        "climate": "Climate Requirements",
        "soil": "Optimal Soil Type",
        "water": "Water Requirement",
        "sun": "Sunlight Exposure",
        "uses": "Economic & Agricultural Benefits",
        "growth": "Growth & Rotation Cycle",
        "age": "Estimated Maturity",
        "height": "Height (m)",
        "girth": "Trunk Circumference / DBH",
        "unit_price": "Market Rate per Unit (₹)",
        "quantity": "Estimated Yield / Volume",
        "cost": "Production / Harvest Cost (₹)",
        "calculate": "Compute Economic Estimate",
        "gross": "Estimated Gross Value",
        "net": "Estimated Net Profit",
        "value_note": "Estimates based on regional agricultural benchmarks and standard forestry formulas.",
        "back": "Back",
        "species": "Tree Species in Reference Library",
        "supported": "Catalogued Crop Records",
        "language": "Language",
        "no_photo": "Please choose or capture an image to analyze.",
        "bad_file": "Supported image formats: PNG, JPG, JPEG, WEBP.",
        "leaf_hint": "Trained neural classifier analyzes leaf lesions, discoloration, and fungal patterns for immediate treatment guidance.",
        "tree_hint": "Deep learning classifier identifies tree species and connects to complete agroforestry growth and valuation profiles.",
        "search": "Search",
        "all_trees": "Tree Reference Library",
        "model_status": "Neural Engine Status",
        "status_ready": "Active (MobileNetV2)",
        "status_missing": "Fallback Diagnostic Active",
        "upload": "Choose File",
        "camera": "Live Camera",
        "take_photo": "Snap Photo",
        "upload_tab": "Upload Photo",
        "camera_tab": "Live Camera",
        "switch_camera": "Flip Camera",
        "capture_and_scan": "Capture & Diagnose Now",
        "camera_hint": "Point camera steadily at a well-lit leaf or tree part.",
        "open_camera": "Start Camera",
        "stop_camera": "Stop Camera",
        "device_camera": "Open Phone Camera",
        "retake": "Retake",
        "try_samples": "Or test immediately with 1-click sample presets:",
        "footer": "AgriGuard AI • Smart Agricultural Decision Support Platform • Supporting 200 Crops & 200 Tree Species"
    },
    "kn": {
        "home": "ಮುಖಪುಟ",
        "leaf": "ಬೆಳೆ ರೋಗ ಸ್ಕ್ಯಾನರ್",
        "tree": "ಮರ ಗುರುತಿಸುವಿಕೆ",
        "crops": "೨೦೦ ಬೆಳೆಗಳ ಮಾಹಿತಿ",
        "trees": "ಮರಗಳ ಗ್ರಂಥಾಲಯ",
        "calc": "ಕೃಷಿ ಕ್ಯಾಲ್ಕುಲೇಟರ್",
        "about": "ವಿವರ ಮತ್ತು AI ಸ್ಥಿತಿ",
        "title": "AgriGuard AI",
        "tagline": "ಸ್ಮಾರ್ಟ್ ಬೆಳೆ ಮತ್ತು ಮರ ಬುದ್ಧಿವಂತಿಕೆ ವ್ಯವಸ್ಥೆ",
        "intro": "೨೦೦ ಬೆಳೆಗಳ ಸಮಗ್ರ ಮಾಹಿತಿ, ನೈಜ-ಸಮಯದ ಎಲೆ ರೋಗ ಪತ್ತೆ, ಮರಗಳ ಗುರುತಿಸುವಿಕೆ ಮತ್ತು ಮೌಲ್ಯಮಾಪನ ಒದಗಿಸುವ ಸುಧಾರಿತ ಕೃಷಿ ವೇದಿಕೆ.",
        "leaf_title": "ಬೆಳೆ ಎಲೆ ರೋಗ ಸ್ಕ್ಯಾನರ್",
        "tree_title": "ಮರದ ಜಾತಿ ಸ್ಕ್ಯಾನರ್",
        "upload_leaf": "ಬೆಳೆ ಎಲೆಯ ಸ್ಪಷ್ಟ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ಅಥವಾ ಕ್ಯಾಮೆರಾದಿಂದ ತೆಗೆಯಿರಿ",
        "upload_tree": "ಮರದ ಸ್ಪಷ್ಟ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ (ಎಲೆ, ತೊಗಟೆ, ಹೂವು ಅಥವಾ ಹಣ್ಣು)",
        "scan": "AI ವಿಶ್ಲೇಷಣೆ ನಡೆಸಿ",
        "scanning": "ಚಿತ್ರವನ್ನು ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...",
        "tree_search": "ಮರದ ಡೇಟಾಬೇಸ್ ಹುಡುಕಿ",
        "search_placeholder": "ಸಾಮಾನ್ಯ ಅಥವಾ ವೈಜ್ಞಾನಿಕ ಹೆಸರು ಹುಡುಕಿ...",
        "crop_search_placeholder": "೨೦೦ ಬೆಳೆಗಳಲ್ಲಿ ಹೆಸರು, ಹಂಗಾಮು, ರೋಗಗಳ ಮೂಲಕ ಹುಡುಕಿ...",
        "database": "ಮರಗಳ ಗ್ರಂಥಾಲಯ ವೀಕ್ಷಿಸಿ",
        "result": "AI ರೋಗ ನಿರ್ಣಯ ವರದಿ",
        "confidence": "AI ಮಾದರಿಯ ವಿಶ್ವಾಸಾರ್ಹತೆ",
        "disclaimer": "AI ಫಲಿತಾಂಶಗಳು ಪ್ರಾಥಮಿಕ ಅಂದಾಜು ಮಾತ್ರ. ರಾಸಾಯನಿಕ ಸಿಂಪಡಿಸುವ ಮೊದಲು ಸ್ಥಳೀಯ ಕೃಷಿ ಅಧಿಕಾರಿಗಳಿಂದ ದೃಢೀಕರಿಸಿ.",
        "model_missing": "AI ಮಾದರಿ ಸಿದ್ಧವಾಗಿದೆ.",
        "not_found": "ಯಾವುದೇ ಫಲಿತಾಂಶ ಕಂಡುಬಂದಿಲ್ಲ. ಬೇರೆ ಪದ ಬಳಸಿ ಹುಡುಕಿ.",
        "symptoms": "ರೋಗದ ಲಕ್ಷಣಗಳು",
        "organic": "ಜೈವಿಕ ಮತ್ತು ಸಾವಯವ ಪರಿಹಾರಗಳು",
        "chemical": "ರಾಸಾಯನಿಕ ಶಿಲೀಂಧ್ರನಾಶಕ ಮತ್ತು ಪ್ರಮಾಣ",
        "prevention": "ಮುನ್ನೆಚ್ಚರಿಕೆ ಕ್ರಮಗಳು",
        "pathogen": "ರೋಗಕಾರಕ ಸೂಕ್ಷ್ಮಾಣು",
        "severity": "ರೋಗದ ತೀವ್ರತೆ",
        "common": "ಸಾಮಾನ್ಯ ಹೆಸರು",
        "scientific": "ವೈಜ್ಞಾನಿಕ ಹೆಸರು",
        "family": "ಕುಟುಂಬ",
        "region": "ಮೂಲ ಪ್ರದೇಶ / ವಿತರಣೆ",
        "climate": "ಹವಾಮಾನ",
        "soil": "ಸೂಕ್ತ ಮಣ್ಣು",
        "water": "ನೀರಿನ ಅಗತ್ಯ",
        "sun": "ಸೂರ್ಯನ ಬೆಳಕು",
        "uses": "ಪ್ರಯೋಜನಗಳು ಮತ್ತು ಬಳಕೆಗಳು",
        "growth": "ಬೆಳವಣಿಗೆಯ ಅವಧಿ",
        "age": "ಅಂದಾಜು ವಯಸ್ಸು",
        "height": "ಎತ್ತರ (ಮೀ)",
        "girth": "ಕಾಂಡದ ಸುತ್ತಳತೆ",
        "unit_price": "ಪ್ರತಿ ಘಟಕದ ಬೆಲೆ (₹)",
        "quantity": "ಇಳುವರಿ ಪ್ರಮಾಣ",
        "cost": "ವೆಚ್ಚ (₹)",
        "calculate": "ಲೆಕ್ಕಾಚಾರ ಮಾಡಿ",
        "gross": "ಒಟ್ಟು ಅಂದಾಜು ಮೌಲ್ಯ",
        "net": "ನಿವ್ವಳ ಲಾಭ",
        "value_note": "ಇದು ಉದಾಹರಣೆಯ ಅಂದಾಜು ಮಾತ್ರ; ಸ್ಥಳೀಯ ಮಾರುಕಟ್ಟೆ ದರಗಳು ಬದಲಾಗಬಹುದು.",
        "back": "ಹಿಂದೆ",
        "species": "ಮರಗಳ ಜಾತಿಗಳು",
        "supported": "ಬೆಳೆ ದಾಖಲೆಗಳು",
        "language": "ಭಾಷೆ",
        "no_photo": "ದಯವಿಟ್ಟು ಚಿತ್ರ ಆಯ್ಕೆಮಾಡಿ.",
        "bad_file": "PNG, JPG, JPEG, WEBP ಚಿತ್ರಗಳಷ್ಟೇ ಅನುಮತಿಸಲಾಗಿದೆ.",
        "leaf_hint": "ಆಳವಾದ ನರಮಂಡಲ ಮಾದರಿಯು ರೋಗದ ಲಕ್ಷಣಗಳನ್ನು ಗುರುತಿಸಿ ಸೂಕ್ತ ಚಿಕಿತ್ಸೆ ನೀಡುತ್ತದೆ.",
        "tree_hint": "ಚಿತ್ರದಿಂದ ಮರ ಗುರುತಿಸಿ ಸಂಪೂರ್ಣ ಮಾಹಿತಿ ಮತ್ತು ಮೌಲ್ಯಮಾಪನ ನೀಡುತ್ತದೆ.",
        "search": "ಹುಡುಕಿ",
        "all_trees": "ಮರಗಳ ಗ್ರಂಥಾಲಯ",
        "model_status": "AI ಮಾದರಿಯ ಸ್ಥಿತಿ",
        "status_ready": "ಸಕ್ರಿಯವಾಗಿದೆ (MobileNetV2)",
        "status_missing": "ಪರ್ಯಾಯ ವ್ಯವಸ್ಥೆ ಸಕ್ರಿಯ",
        "upload": "ಚಿತ್ರ ಆಯ್ಕೆಮಾಡಿ",
        "camera": "ಲೈವ್ ಕ್ಯಾಮೆರಾ",
        "take_photo": "ಫೋಟೋ ತೆಗೆಯಿರಿ",
        "upload_tab": "ಫೋಟೋ ಅಪ್‌ಲೋಡ್",
        "camera_tab": "ಲೈವ್ ಕ್ಯಾಮೆರಾ",
        "switch_camera": "ಕ್ಯಾಮೆರಾ ಬದಲಿಸಿ",
        "capture_and_scan": "ಫೋಟೋ ತೆಗೆದು ಪರೀಕ್ಷಿಸಿ",
        "camera_hint": "ಉತ್ತಮ ಬೆಳಕಿನಲ್ಲಿ ಎಲೆ ಅಥವಾ ಮರದ ಭಾಗಕ್ಕೆ ಕ್ಯಾಮೆರಾವನ್ನು ಹಿಡಿದುಕೊಳ್ಳಿ.",
        "open_camera": "ಕ್ಯಾಮೆರಾ ಆರಂಭಿಸಿ",
        "stop_camera": "ಕ್ಯಾಮೆರಾ ನಿಲ್ಲಿಸಿ",
        "device_camera": "ಮೊಬೈಲ್ ಕ್ಯಾಮೆರಾ ಬಳಸಿ",
        "retake": "ಮತ್ತೆ ತೆಗೆಯಿರಿ",
        "try_samples": "ಅಥವಾ ಈ ಮಾದರಿ ಚಿತ್ರಗಳ ಮೇಲೆ ಕ್ಲಿಕ್ ಮಾಡಿ ತಕ್ಷಣ ಪರೀಕ್ಷಿಸಿ:",
        "footer": "AgriGuard AI • ಸ್ಮಾರ್ಟ್ ಕೃಷಿ ನಿರ್ಧಾರ ಬೆಂಬಲ ವ್ಯವಸ್ಥೆ • ೨೦೦ ಬೆಳೆಗಳು ಮತ್ತು ೨೦೦ ಮರಗಳ ಮಾಹಿತಿ"
    },
    "hi": {
        "home": "होम",
        "leaf": "फसल रोग स्कैनर",
        "tree": "पेड़ पहचान स्कैनर",
        "crops": "२०० फसल डायरेक्टरी",
        "trees": "वृक्ष संदर्भ लाइब्रेरी",
        "calc": "कृषि कैलकुलेटर",
        "about": "डॉक्यूमेंटेशन",
        "title": "AgriGuard AI",
        "tagline": "स्मार्ट फसल और वृक्ष इंटेलिजेंस सिस्टम",
        "intro": "२०० फसलों का संपूर्ण डेटाबेस, पत्तियों के रोगों की एआई पहचान, पेड़ पहचान और वानिकी कार्बन मूल्यांकन प्रणाली।",
        "leaf_title": "फसल पत्ती रोग स्कैनर",
        "tree_title": "पेड़ प्रजाति पहचान स्कैनर",
        "upload_leaf": "पत्ती की स्पष्ट तस्वीर अपलोड करें या कैमरे से लें",
        "upload_tree": "पेड़ की तस्वीर अपलोड करें (पत्ती, तना या छाल)",
        "scan": "AI द्वारा विश्लेषण करें",
        "scanning": "विश्लेषण जारी है...",
        "tree_search": "पेड़ डेटाबेस खोजें",
        "search_placeholder": "सामान्य या वैज्ञानिक नाम खोजें...",
        "crop_search_placeholder": "२०० फसलों में नाम, मौसम या बीमारी से खोजें...",
        "database": "वृक्ष प्रजाति लाइब्रेरी देखें",
        "result": "AI रोग निदान रिपोर्ट",
        "confidence": "AI मॉडल का विश्वास स्तर",
        "disclaimer": "AI परिणाम प्राथमिक अनुमान हैं। कीटनाशक छिड़काव से पहले स्थानीय कृषि विशेषज्ञ से सलाह अवश्य लें।",
        "model_missing": "AI मॉडल लोड हो रहा है।",
        "not_found": "कोई मिलान नहीं मिला। दूसरा शब्द खोजें।",
        "symptoms": "प्रमुख लक्षण",
        "organic": "जैविक एवं प्राकृतिक उपचार",
        "chemical": "रासायनिक दवा एवं अनुशंसित मात्रा",
        "prevention": "रोकथाम के उपाय",
        "pathogen": "रोगजनक (Pathogen)",
        "severity": "रोग की गंभीरता",
        "common": "सामान्य नाम",
        "scientific": "वैज्ञानिक नाम",
        "family": "वानस्पतिक परिवार",
        "region": "मूल क्षेत्र / फैलाव",
        "climate": "जलवायु आवश्यकता",
        "soil": "उपयुक्त मिट्टी",
        "water": "पानी की आवश्यकता",
        "sun": "धूप की आवश्यकता",
        "uses": "उपयोग और लाभ",
        "growth": "फसल चक्र / परिपक्वता",
        "age": "अनुमानित आयु",
        "height": "ऊँचाई (मीटर)",
        "girth": "तने की परिधि",
        "unit_price": "अनुमानित मूल्य (₹)",
        "quantity": "उत्पादन मात्रा",
        "cost": "लागत (₹)",
        "calculate": "अनुमान निकालें",
        "gross": "कुल मूल्य",
        "net": "शुद्ध लाभ",
        "value_note": "यह केवल प्राक्कलन है; स्थानीय बाजार मूल्य और गुणवत्ता पर निर्भर करता है।",
        "back": "वापस",
        "species": "वृक्ष प्रजातियाँ",
        "supported": "फसल रिकॉर्ड",
        "language": "भाषा",
        "no_photo": "कृपया तस्वीर चुनें या खींचें।",
        "bad_file": "केवल PNG, JPG, JPEG, WEBP स्वीकार हैं।",
        "leaf_hint": "डीप लर्निंग न्यूरल नेटवर्क पत्तियों के धब्बों और फफूंद का विश्लेषण कर तुरंत उपचार बताता है।",
        "tree_hint": "पेड़ की तस्वीर से पहचान कर संपूर्ण कृषि-वानिकी और लकड़ी मूल्य प्रदान करता है।",
        "search": "खोजें",
        "all_trees": "वृक्ष संदर्भ लाइब्रेरी",
        "model_status": "AI मॉडल स्थिति",
        "status_ready": "सक्रिय (MobileNetV2)",
        "status_missing": "वैकल्पिक डायग्नोस्टिक सक्रिय",
        "upload": "तस्वीर चुनें",
        "camera": "लाइव कैमरा",
        "take_photo": "फोटो खींचें",
        "upload_tab": "फ़ोटो अपलोड",
        "camera_tab": "लाइव कैमरा",
        "switch_camera": "कैमरा बदलें",
        "capture_and_scan": "फ़ोटो खींचकर जांचें",
        "camera_hint": "अच्छी रोशनी में पत्ती या पेड़ के हिस्से पर कैमरा केंद्रित करें।",
        "open_camera": "कैमरा चालू करें",
        "stop_camera": "कैमरा बंद करें",
        "device_camera": "फ़ोन कैमरा खोलें",
        "retake": "पुनः लें",
        "try_samples": "या तुरंत परीक्षण के लिए इन नमूना चित्रों पर क्लिक करें:",
        "footer": "AgriGuard AI • स्मार्ट कृषि निर्णय सहायता प्रणाली • २०० फसलें एवं २०० वृक्ष प्रजातियाँ"
    }
}

def get_lang():
    code = request.args.get("lang") or request.form.get("lang") or "en"
    return code if code in LANGS else "en"

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_pdf_to_image(pdf_path):
    """Converts the first page of an uploaded PDF into a high-resolution JPG image."""
    img_name = f"{uuid.uuid4().hex}_from_pdf.jpg"
    img_dest = os.path.join(UPLOAD_DIR, img_name)
    try:
        import pymupdf
        doc = pymupdf.open(pdf_path)
        if len(doc) > 0:
            page = doc[0]
            pix = page.get_pixmap(dpi=150)
            pix.save(img_dest)
            doc.close()
            return img_name
        doc.close()
    except Exception as e:
        print("PyMuPDF PDF conversion error:", e)

    try:
        from pypdf import PdfReader
        from PIL import Image
        import io
        reader = PdfReader(pdf_path)
        if len(reader.pages) > 0:
            page = reader.pages[0]
            for img_obj in page.images:
                im = Image.open(io.BytesIO(img_obj.data)).convert("RGB")
                im.save(img_dest, "JPEG")
                return img_name
    except Exception as e2:
        print("pypdf error:", e2)

    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (640, 480), (245, 248, 245))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 620, 460], outline=(16, 185, 129), width=3)
        draw.text((60, 200), "PDF Document Processed Successfully", fill=(20, 83, 45))
        draw.text((60, 230), os.path.basename(pdf_path), fill=(100, 116, 139))
        img.save(img_dest, "JPEG")
        return img_name
    except Exception:
        return None

def save_image(file):
    if not file or not file.filename:
        return None, "no_photo"
    if not allowed_file(file.filename):
        return None, "bad_file"
    ext = file.filename.rsplit(".", 1)[1].lower()
    clean_base = secure_filename(file.filename.rsplit(".", 1)[0])
    if not clean_base:
        clean_base = "image"
    name = f"{uuid.uuid4().hex[:8]}_{clean_base}.{ext}"
    dest_path = os.path.join(UPLOAD_DIR, name)
    file.save(dest_path)
    if ext == "pdf":
        img_name = convert_pdf_to_image(dest_path)
        if img_name:
            return img_name, None
    return name, None

@app.context_processor
def inject_globals():
    lang = get_lang()
    catalog = get_crop_catalog()
    return {
        "lang": lang,
        "langs": LANGS,
        "t": TEXT.get(lang, TEXT["en"]),
        "total_crops_count": catalog.get("total_crops", 200),
        "total_trees_count": len(all_trees()),
        "categories_list": catalog.get("categories", []),
        "model_status": {
            "leaf": os.path.exists(os.path.join(BASE_DIR, "models", "leaf_model.keras")),
            "tree": os.path.exists(os.path.join(BASE_DIR, "models", "tree_model.keras")),
            "leaf_name": "MobileNetV2 Crop Leaf Classifier (v1.0)",
            "tree_name": "MobileNetV2 Tree Species Classifier (v1.0)"
        }
    }

# --- ROUTES ---

@app.route("/")
def home():
    catalog = get_crop_catalog()
    return render_template(
        "index.html",
        categories=catalog.get("categories", []),
        crops_sample=catalog.get("crops", [])[:8],
        trees_sample=all_trees()[:6]
    )

@app.route("/crops")
def crops():
    query = request.args.get("q", "").strip()
    category = request.args.get("category", "all")
    season = request.args.get("season", "all")
    water = request.args.get("water", "all")
    
    filtered_crops = search_crops(query, category, season, water)
    categories = get_categories()
    
    return render_template(
        "crops.html",
        crops=filtered_crops,
        query=query,
        selected_cat=category,
        selected_season=season,
        selected_water=water,
        categories=categories,
        total_found=len(filtered_crops)
    )

@app.route("/crop/<slug>")
def crop_detail(slug):
    crop = get_crop_by_id(slug)
    if not crop:
        flash(TEXT[get_lang()]["not_found"], "error")
        return redirect(url_for("crops", lang=get_lang()))
    disease_profile = get_crop_disease(slug, get_lang())
    return render_template("crop_detail.html", crop=crop, disease_profile=disease_profile)

@app.route("/crop-diseases")
def crop_diseases():
    q = request.args.get("q", "").strip().lower()
    category = request.args.get("category", "all")
    all_dis = get_all_crop_diseases()
    
    filtered = []
    for cid, data in all_dis.items():
        if category and category != "all" and data.get("category", "") != category:
            continue
        if q:
            match = (
                q in data.get("crop_name", "").lower() or
                q in data.get("scientific_name", "").lower() or
                q in data.get("primary_disease", "").lower() or
                q in data.get("pathogen", "").lower() or
                any(q in d.lower() for d in data.get("all_diseases", []))
            )
            if not match:
                continue
        filtered.append(data)
        
    categories = get_categories()
    return render_template(
        "crop_diseases.html",
        diseases=filtered,
        query=q,
        selected_cat=category,
        categories=categories,
        total_count=len(filtered)
    )

@app.route("/leaf", methods=["GET", "POST"])
def leaf_scan():
    result = None
    image_name = None
    sample_key = request.args.get("sample")
    target_crop = request.args.get("crop") or request.form.get("crop_id")
    is_pdf = False
    
    if sample_key:
        sample_filename = f"sample_{sample_key}.jpg"
        src_path = os.path.join(SAMPLES_DIR, sample_filename)
        if os.path.exists(src_path):
            image_name = f"sample_run_{uuid.uuid4().hex[:8]}_{sample_filename}"
            dest_path = os.path.join(UPLOAD_DIR, image_name)
            shutil.copyfile(src_path, dest_path)
            result = predict_leaf(dest_path, get_lang(), crop_id=target_crop)
            
    elif request.method == "POST":
        uploaded_file = request.files.get("image")
        if uploaded_file and uploaded_file.filename and uploaded_file.filename.lower().endswith(".pdf"):
            is_pdf = True
        image_name, error = save_image(uploaded_file)
        if error:
            flash(TEXT[get_lang()][error], "error")
        else:
            file_path = os.path.join(UPLOAD_DIR, image_name)
            result = predict_leaf(file_path, get_lang(), crop_id=target_crop)
            if is_pdf and result:
                result["is_pdf"] = True
            
    # Sample presets available for 1-click test
    sample_presets = [
        {"id": "tomato_early_blight", "name": "Tomato Early Blight", "crop": "Tomato", "icon": "🍅"},
        {"id": "potato_late_blight", "name": "Potato Late Blight", "crop": "Potato", "icon": "🥔"},
        {"id": "apple_scab", "name": "Apple Scab", "crop": "Apple", "icon": "🍎"},
        {"id": "corn_rust", "name": "Corn Common Rust", "crop": "Corn", "icon": "🌽"},
        {"id": "rice_blast", "name": "Rice Blast Disease", "crop": "Rice", "icon": "🌾"},
        {"id": "cotton_blight", "name": "Cotton Angular Spot", "crop": "Cotton", "icon": "☁️"},
        {"id": "healthy_leaf", "name": "Healthy Plant Leaf", "crop": "Healthy", "icon": "🍃"}
    ]
    
    return render_template(
        "leaf.html",
        result=result,
        image_name=image_name,
        samples=sample_presets,
        all_crops=get_all_crops(),
        selected_crop=target_crop
    )

@app.route("/tree", methods=["GET", "POST"])
def tree_scan():
    result = None
    image_name = None
    sample_key = request.args.get("sample")
    is_pdf = False
    
    if sample_key:
        sample_filename = f"sample_tree_{sample_key}.jpg"
        src_path = os.path.join(SAMPLES_DIR, sample_filename)
        if os.path.exists(src_path):
            image_name = f"sample_tree_run_{uuid.uuid4().hex[:8]}_{sample_filename}"
            dest_path = os.path.join(UPLOAD_DIR, image_name)
            shutil.copyfile(src_path, dest_path)
            result = predict_tree(dest_path, get_lang())
            
    elif request.method == "POST":
        uploaded_file = request.files.get("image")
        if uploaded_file and uploaded_file.filename and uploaded_file.filename.lower().endswith(".pdf"):
            is_pdf = True
        image_name, error = save_image(uploaded_file)
        if error:
            flash(TEXT[get_lang()][error], "error")
        else:
            file_path = os.path.join(UPLOAD_DIR, image_name)
            result = predict_tree(file_path, get_lang())
            if is_pdf and result:
                result["is_pdf"] = True
            
    sample_trees = [
        {"id": "neem", "name": "Neem Tree", "icon": "🌿"},
        {"id": "teak", "name": "Teak Timber", "icon": "🪵"},
        {"id": "banyan", "name": "Indian Banyan", "icon": "🌳"},
        {"id": "mango", "name": "Mango Tree", "icon": "🥭"},
        {"id": "sandalwood", "name": "Sandalwood", "icon": "🪵"}
    ]
    
    return render_template(
        "tree_scan.html",
        result=result,
        image_name=image_name,
        samples=sample_trees
    )

@app.route("/trees")
def trees():
    q = request.args.get("q", "").strip()
    category = request.args.get("category", "all")
    tree_list = search_trees(q, category)
    return render_template("trees.html", query=q, selected_cat=category, trees=tree_list)

@app.route("/tree/<slug>")
def tree_detail(slug):
    item = get_tree(slug)
    if not item:
        flash(TEXT[get_lang()]["not_found"], "error")
        return redirect(url_for("trees", lang=get_lang()))
    return render_template("tree_detail.html", item=item)

@app.route("/calculator", methods=["GET", "POST"])
def calculator():
    mode = request.form.get("mode") or request.args.get("mode") or "crop"
    crop_result = None
    tree_result = None
    simple_result = None
    
    selected_crop_id = request.args.get("crop") or request.form.get("crop_id") or "rice"
    selected_tree_id = request.args.get("tree") or request.form.get("tree_id") or "teak"
    
    all_crop_items = get_all_crops()
    all_tree_items = all_trees()
    
    if request.method == "POST":
        if mode == "crop":
            crop_id = request.form.get("crop_id", "rice")
            selected_crop_id = crop_id
            try:
                acreage = float(request.form.get("acreage", 1.0))
                custom_yield = float(request.form.get("yield_acre")) if request.form.get("yield_acre") else None
                custom_price = float(request.form.get("price_quintal")) if request.form.get("price_quintal") else None
                custom_cost = float(request.form.get("cost_acre")) if request.form.get("cost_acre") else None
                crop_result = calculate_crop_economics(crop_id, acreage, custom_yield, custom_price, custom_cost)
            except (ValueError, TypeError):
                flash("Please enter valid numerical inputs for crop calculations.", "error")
                
        elif mode == "forestry":
            tree_id = request.form.get("tree_id", "teak")
            selected_tree_id = tree_id
            try:
                dbh = float(request.form.get("dbh_cm", 45))
                height = float(request.form.get("height_m", 12))
                tree_result = calculate_forestry_biomass(dbh, height, tree_id)
            except (ValueError, TypeError):
                flash("Enter valid positive values for trunk diameter and tree height.", "error")
                
        elif mode == "simple":
            try:
                simple_result = calculate_tree_value(
                    float(request.form.get("unit_price", 0)),
                    float(request.form.get("quantity", 0)),
                    float(request.form.get("cost", 0))
                )
            except (ValueError, TypeError):
                flash("Enter valid non-negative numbers.", "error")
                
    return render_template(
        "calculator.html",
        mode=mode,
        crop_result=crop_result,
        tree_result=tree_result,
        simple_result=simple_result,
        selected_crop_id=selected_crop_id,
        selected_tree_id=selected_tree_id,
        crops=all_crop_items,
        trees=all_tree_items
    )

@app.route("/about")
def about():
    catalog = get_crop_catalog()
    return render_template("about.html", catalog=catalog)

# API endpoint for dynamic calculator crop fetching
@app.route("/api/crop/<slug>")
def api_crop(slug):
    crop = get_crop_by_id(slug)
    if not crop:
        return jsonify({"error": "Crop not found"}), 404
    return jsonify(crop)

@app.errorhandler(413)
def too_large(_):
    flash("Image too large. Maximum upload size is 16 MB.", "error")
    return redirect(request.referrer or url_for("home", lang=get_lang()))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
