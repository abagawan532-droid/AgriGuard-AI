# AgriGuard AI — complete build and run guide

## 1. What this project includes
- Flask website with separate crop-leaf and tree image upload workflows.
- Interface text in English, Kannada and Hindi.
- Optional TensorFlow/Keras image model loading (`models/leaf_model.keras` and `models/tree_model.keras`).
- Tree reference database, tree profile pages and search.
- Transparent tree gross/net estimate calculator.
- Training scripts, project report, test plan and architecture notes.

**Important:** This source package does not include trained model weights or an exhaustive worldwide tree database. Those require training data, licenses, training time and evaluation. The app deliberately does not invent image predictions when weights are missing.

## 2. Windows / VS Code setup
Install Python 3.10 or 3.11 (64-bit) and VS Code. Open this folder in VS Code. In Terminal > New Terminal, run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

If PowerShell blocks activation, use Command Prompt terminal and run:
```bat
.venv\Scripts\activate.bat
```

Open `http://127.0.0.1:5000` in your browser. Stop the server with Ctrl+C.

TensorFlow installation depends on your operating system, Python version and hardware. If it fails, first install Flask, Pillow and NumPy (`pip install Flask Pillow numpy`) to test the website; image models require a compatible TensorFlow installation.

## 3. Add/train a crop disease model
1. Obtain a legally usable and clearly labelled leaf-disease dataset. PlantVillage is one commonly used research dataset; check its current license/terms and dataset scope before use. A model trained on a limited set of crops/classes cannot diagnose all crops.
2. Make folders:
```
datasets/leaf/train/<class_name>/
datasets/leaf/validation/<class_name>/
```
3. Split by source plant/image session, not just random near-duplicates. Keep an independent test set outside training and validation.
4. Put images into folders named exactly after the class. Example: `Apple___healthy`, `Apple___Apple_scab`.
5. Run:
```
python train_leaf_model.py
```
6. Check validation and independent test metrics, confusion matrix, false negatives and performance on phone photos. Do not deploy a model based only on training accuracy.
7. The script saves `models/leaf_model.keras` and writes labels to `data/leaf_labels.json`.
8. Restart the Flask app. It will show predictions if the model loads successfully.

The provided sample `leaf_labels.json` is only a label example. It is replaced by the actual dataset's class list during training.

## 4. Add/train a tree recognition model
1. Obtain a properly licensed, diverse tree image dataset. For global coverage, the dataset must cover many species, seasons, regions, camera angles and stages. Avoid scraping images without checking rights.
2. Make folders:
```
datasets/tree/train/<species_id>/
datasets/tree/validation/<species_id>/
```
3. Directory names should match tree `id` values in `data/tree_species.json` if you want automatic profile matching.
4. Add enough examples per class and keep a separate independent test set.
5. Run `python train_tree_model.py`.
6. Evaluate top-1/top-5 accuracy and per-species recall, including unknown species and look-alike species. A closed-set classifier will force an image into one of its known classes; use a confidence threshold and unknown-class strategy before public deployment.
7. The script saves `models/tree_model.keras` and writes labels to `data/tree_labels.json`.

The sample tree data has 10 records. It is a demonstration database, not “all trees in the world”. For broad global coverage, consider a taxonomic backbone such as World Flora Online or Plants of the World Online and a licensed image source such as GBIF occurrence media or iNaturalist observations, following each source's API terms and per-image license. Keep attribution/license metadata for each image and record. Validate taxonomy and regional distribution before publishing.

## 5. Tree age and value
A single image cannot reliably provide exact age. Better estimates require known planting dates, local growth curves, species, trunk diameter measured at a standard height, site conditions and sometimes forestry expertise. Tree-ring methods can be invasive and are not appropriate for every species.
The calculator intentionally uses values you enter: gross = unit price × quantity; net = gross − cost. It does not claim that price or saleable quantity can be inferred from a picture. For real estimates, use current local market quotes and confirm permits/protected-tree rules with the relevant authority.

## 6. Change language
Use the language dropdown on the website: English, ಕನ್ನಡ, हिन्दी. The UI labels are translated. Tree reference facts are currently stored in English; translate data fields before claiming full multilingual scientific content.

## 7. Security and deployment
- Change `AGRIGUARD_SECRET` before deployment.
- Do not deploy Flask's debug server publicly.
- Add authentication, upload cleanup, rate limits, logging and secure production hosting for public use.
- Images are saved in `static/uploads`; delete them periodically if you use real people's or farm images.
- Run only with images you have permission to process.

## 8. Troubleshooting
- `ModuleNotFoundError: flask`: activate `.venv`, then `pip install -r requirements.txt`.
- Model missing: train/copy the model into the exact `models` path and ensure TensorFlow is compatible.
- Model output classes mismatch: retrain so `*_labels.json` matches the saved model's output order.
- Blank/poor predictions: check image quality, preprocessing and model evaluation; do not trust low-confidence predictions.
- Tree profile not found: ensure the predicted label exactly matches a tree `id` in `data/tree_species.json`.
