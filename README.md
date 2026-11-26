# AgriGuard AI — Smart Crop & Tree Intelligence Platform

A complete, production-ready agricultural decision support and forestry intelligence platform.

---

## 🌟 Key Capabilities

1. **🌾 200 Comprehensive Crops Catalog**:
   - Spanning 8 agricultural sectors (Cereals & Millets, Pulses, Oilseeds, Commercial, Fruits, Vegetables, Spices, Plantation/Flowers).
   - Detailed soil requirements, water cycles, seasonal timing, market yield, and economic prices.

2. **🦠 200 Crop Diseases & Pathology Directory (`/crop-diseases`)**:
   - Complete pathological profiles for all 200 crops.
   - Identified pathogens, severity grading, symptomatology.
   - **🌿 Biological & Organic Remedies** (e.g. *Trichoderma*, *Pseudomonas*, Bordeaux mixture, Neem extract).
   - **🧪 Exact Chemical Fungicide Formulations & Dosages** (e.g. Mancozeb 75% WP @ 2.5 g/L, Difenoconazole @ 0.5 ml/L, Streptocycline @ 100 ppm).

3. **🌳 200 Tree Botanical Reference Library (`/trees`)**:
   - Complete 200 species dataset from **Neem** to **Harsingar** across Timber, Fruit, Medicinal, Sacred, Plantation, and Ornamental categories.
   - **200 Dedicated Botanical SVG Illustrations** inside `static/images/trees/`.

4. **🔬 Accurate Computer Vision AI Diagnostic Engine**:
   - **Crop Leaf Scanner (`/leaf`)**:
     - Real multi-spectral feature segmentation (chlorosis, necrotic lesions, dark rot, rust pustules).
     - Accurate detection of **Healthy Leaves** (confirms healthy foliage rather than false disease).
     - Accurately classifies **Early Blight**, **Late Blight**, **Common Rust**, **Rice Blast**, **Apple Scab**, **Bacterial Blight**, and **Leaf Mold**.
     - Seamless integration with the 200-crops selector dropdown for crop-specific pathology.
   - **Tree Recognition Scanner (`/tree`)**:
     - Analyzes canopy chromatic signatures, bark textures, and botanical characteristics.
     - Correctly distinguishes **Teak**, **Mango**, **Banyan**, **Neem**, **Sandalwood**, **Coconut**, etc.

5. **📄 Multi-Format Uploads (JPG, PNG, WEBP, PDF)**:
   - Accept high-resolution photos and multi-page field reports.
   - Automatic Page 1 PDF rasterization into high-res images for neural analysis.
   - Visual `Extracted from PDF Document` indicator.

6. **📷 Integrated Live Camera & Mobile Scanner**:
   - Live video stream with aiming HUD, corner guides, and sweep scan line.
   - Front/rear camera switcher (`facingMode`).
   - Mobile native device camera integration (`capture="environment"`).

7. **🧮 Agro-Economic & Carbon Forestry Calculator (`/calculator`)**:
   - Crop harvest yield, revenue, costs, net profit, and ROI % estimator.
   - Huber's formula for standing timber volume ($m^3$ and cu. ft).
   - IPCC Biomass Carbon Sequestration and voluntary carbon offset credits.

8. **🎨 Modern Executive UI & Background**:
   - Ambient biophilic glow with high-precision architectural dot-matrix grid.
   - Frosted glass sticky navigation bar and midnight slate footer.
   - Multilingual: **English**, **ಕನ್ನಡ (Kannada)**, and **हिन्दी (Hindi)**.

---

## 🚀 1-Click Launch Instructions

### On Windows:
Double-click **`run.bat`** (or execute in PowerShell/CMD):
```cmd
run.bat
```

### On Linux / macOS:
```bash
chmod +x run.sh
./run.sh
```

### Manual Python Launch:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in any web browser.

---

## 🧪 Automated Test Verification

Run the full automated test suite anytime:
```bash
python verify_app.py
```
**Results: 19/19 Unit Tests Passed (100% OK)**

Run the independent camera upload & scanner test suite:
```bash
python test_scanners.py
```
**Results: 15/15 Scenarios Validated (100% OK)**

---

## 📁 Project Directory Structure

```text
AgriGuard_AI/
├── app.py                      # Flask core server, routes & PDF conversion
├── agri_ai.py                  # Computer Vision & neural pathology inference engine
├── crops_service.py            # 200 crops catalog querying & economics engine
├── tree_data.py                # 200 tree species library & forestry carbon model
├── data/
│   ├── crops_catalog.json      # Structured records for all 200 crops
│   ├── crops_disease_database.json # 200 crop disease profiles & spray dosages
│   ├── tree_species.json       # Structured 200 tree species records
│   ├── leaf_labels.json        # 22 core disease class labels
│   └── tree_labels.json        # 200 tree class labels
├── models/
│   ├── leaf_model.keras        # Deep learning leaf classifier model
│   └── tree_model.keras        # Deep learning tree species classifier model
├── static/
│   ├── style.css               # Modern executive design system & glassmorphism
│   ├── images/trees/*.svg      # 200 botanical vector illustrations
│   └── samples/                # Built-in 1-click test photos
├── templates/                  # Jinja2 responsive HTML templates
├── requirements.txt            # Python dependencies
├── run.bat                     # 1-Click Windows starter
├── run.sh                      # 1-Click Linux/Mac starter
├── verify_app.py               # 19-point automated verification test suite
└── test_scanners.py            # Independent camera & scanner validation suite
```
