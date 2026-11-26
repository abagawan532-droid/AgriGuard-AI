# AgriGuard AI — Complete Project Package

This package preserves the source files from the uploaded AgriGuard_AI_Project ZIP and adds
empty dataset folder structure for preparing both model pipelines.

## Included
- Flask application and existing HTML/CSS templates
- AI helper and model-training scripts
- Project report, architecture, build guide, and test plan
- Existing JSON metadata and label files
- Dataset folder placeholders for crop and tree train/validation images

## Important model status
This ZIP does NOT contain trained model weights or actual image datasets.
- `models/leaf_model.keras`: not included; crop model still needs training.
- `models/tree_model.keras`: not included in the uploaded ZIP; the model seen on the
  user's laptop is separate and should be backed up before any replacement.
- The empty dataset folders are placeholders, not images.

A model must be trained using real, correctly labelled images and evaluated on a
validation/test set before it is described as trained or reliable.

## Dataset folder layout
Crop:
- `datasets/leaf/train/<class_name>/`
- `datasets/leaf/validation/<class_name>/`

Tree:
- `datasets/tree/train/<species_name>/`
- `datasets/tree/validation/<species_name>/`

For each task, training and validation must use matching class/species folder names.
Do not put the same image in both training and validation sets.

## Run the existing Flask app
Create/activate a virtual environment, install `requirements.txt`, then run:
`python app.py`

The website may still report that a model is not installed until the trained `.keras`
files and their matching label files are generated and placed in `models/`.
