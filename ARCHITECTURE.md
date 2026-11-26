# Architecture

```text
User browser
    |
    v
Flask web application (app.py)
    |-- /leaf --> image validation/upload --> agri_ai.predict_leaf()
    |                                      --> optional leaf_model.keras
    |-- /tree --> image validation/upload --> agri_ai.predict_tree()
    |                                      --> optional tree_model.keras
    |                                      --> tree profile lookup
    |-- /trees and /tree/<id> --> tree_data.py --> data/tree_species.json
    |-- /calculator --> user-entered price, quantity, costs
    |
    v
Jinja templates + responsive CSS + English/Kannada/Hindi UI labels
```

Model weights are deliberately separate from source code and must be trained/validated. The reference data is illustrative and should be expanded with verified, licensed information.
