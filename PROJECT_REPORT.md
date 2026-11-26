# AgriGuard AI — Smart Crop & Tree Intelligence System
## Mini-project report

### Abstract
AgriGuard AI is a software-based decision-support prototype with two intended image-analysis workflows: crop-leaf disease classification and tree-species recognition. The application provides a multilingual web interface, a searchable tree reference library, and a transparent arithmetic tool for estimating gross and net value from user-supplied assumptions. Image predictions are enabled only when compatible trained models are installed. The system is designed to avoid fabricating predictions when models are unavailable.

### Problem statement
Farmers, students and land managers may have difficulty accessing quick, understandable information about visible crop-leaf symptoms and tree species. Online information can be scattered, technical, or not available in the user's preferred language. A single software interface can help organize preliminary information, while preserving the need for expert verification.

### Objectives
1. Build a responsive website with crop and tree scanner workflows.
2. Support English, Kannada and Hindi interface text.
3. Provide crop disease class predictions when a trained model is available.
4. Provide tree species predictions when a trained model is available.
5. Present tree distribution, growing conditions, uses, maturity notes and age/value limitations.
6. Provide a searchable reference database and a transparent estimate calculator.
7. Document training, testing, limitations and safe use.

### Scope
The supplied reference database contains a small set of example species and is not a complete global taxonomy. The supplied source code includes transfer-learning training scripts but does not bundle trained weights. Broad global recognition requires a much larger licensed dataset and evaluation. The system is educational decision support, not a replacement for agricultural extension, a botanist, or a forestry professional.

### System architecture
Browser → Flask routes/templates → image upload validation → optional TensorFlow/Keras model loader → predicted class label → tree profile lookup or disease guidance. Tree search reads the JSON reference library. The calculator performs gross = unit price × quantity and net = gross − costs.

### Technologies
- Python
- Flask
- HTML/CSS
- TensorFlow/Keras (optional trained image models)
- Pillow and NumPy
- JSON reference data

### Modules
1. Home and language selection
2. Crop leaf image upload and model inference
3. Tree image upload and model inference
4. Tree reference search and details
5. Tree gross/net estimate calculator
6. Model training scripts and build guide

### Methodology
The model training scripts use MobileNetV2 transfer learning with a frozen feature extractor and a small softmax classification head. Dataset folders supply the class labels. A proper evaluation should include independent test images, per-class precision/recall, confusion matrices, robustness tests across lighting/backgrounds, and checks for out-of-distribution images. Model confidence is not equivalent to medical/agronomic certainty.

### Testing plan
- Home page loads and language selection works.
- Leaf and tree routes accept supported image types and reject unsupported files.
- Upload size is limited.
- With no weights, the app displays an honest missing-model message.
- With compatible weights, inference returns a label and confidence.
- Tree database search works by common name, scientific name, family or alternate name.
- Tree details show distribution, climate, soil, water, sunlight, uses and growth notes.
- Calculator correctly computes gross and net values and rejects negative numbers.
- Test mobile layout and keyboard navigation.
- Verify translations with fluent speakers.
- Evaluate models on independent, licensed images before making performance claims.

### Ethical, safety and legal considerations
Wrong disease predictions can cause crop losses or unsafe pesticide use. All disease treatment should be confirmed by a local agricultural expert. Dataset image licenses and attribution requirements must be respected. Tree sale and removal may be regulated. Do not claim exact tree age or guaranteed profit from an image.

### Future work
- Add verified multilingual scientific facts.
- Expand the global taxonomy and licensed image set.
- Add model calibration and unknown-species rejection.
- Use location/climate context only with user consent and reliable sources.
- Add explainability, data provenance, image cleanup and a deployment security review.
- Evaluate with farmers, forestry experts and multilingual users.

### Conclusion
AgriGuard AI provides a runnable web application foundation and documented path for adding trained crop and tree classifiers. Its value depends on dataset quality, model evaluation, current reference information and responsible use.
