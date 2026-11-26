import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import tensorflow as tf
from tensorflow.keras import layers, models

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("static/samples", exist_ok=True)

# 1. Update leaf_labels.json
leaf_labels = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Cotton___Bacterial_blight",
    "Cotton___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Rice___Bacterial_blight",
    "Rice___Blast",
    "Rice___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___healthy",
    "Wheat___Yellow_rust",
    "Wheat___healthy"
]

with open("data/leaf_labels.json", "w", encoding="utf-8") as f:
    json.dump(leaf_labels, f, ensure_ascii=False, indent=2)
print(f"Saved {len(leaf_labels)} leaf labels to data/leaf_labels.json")

# 2. Update tree_labels.json (matching 15 tree species)
with open("data/tree_species.json", "r", encoding="utf-8") as f:
    trees = json.load(f)
tree_labels = [t["id"] for t in trees]

with open("data/tree_labels.json", "w", encoding="utf-8") as f:
    json.dump(tree_labels, f, ensure_ascii=False, indent=2)
print(f"Saved {len(tree_labels)} tree labels to data/tree_labels.json")

# 3. Build & save models/leaf_model.keras
print("Building models/leaf_model.keras...")
inputs_leaf = layers.Input(shape=(224, 224, 3), name="leaf_input")
x = layers.Rescaling(1./255)(inputs_leaf)
base_leaf = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
x = base_leaf(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs_leaf = layers.Dense(len(leaf_labels), activation="softmax", name="leaf_predictions")(x)
model_leaf = models.Model(inputs_leaf, outputs_leaf, name="AgriGuard_Leaf_Classifier")
model_leaf.save("models/leaf_model.keras")
print("Saved models/leaf_model.keras successfully!")

# 4. Build & save models/tree_model.keras
print("Building models/tree_model.keras...")
inputs_tree = layers.Input(shape=(224, 224, 3), name="tree_input")
x = layers.Rescaling(1./255)(inputs_tree)
base_tree = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
x = base_tree(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs_tree = layers.Dense(len(tree_labels), activation="softmax", name="tree_predictions")(x)
model_tree = models.Model(inputs_tree, outputs_tree, name="AgriGuard_Tree_Classifier")
model_tree.save("models/tree_model.keras")
print("Saved models/tree_model.keras successfully!")

# 5. Generate realistic Agronomic Sample Images for 1-Click Testing in static/samples/
def create_sample_leaf(filename, main_color, spot_color=None, spots=0, rust=False, blight=False):
    img = Image.new("RGB", (400, 400), (242, 246, 242))
    draw = ImageDraw.Draw(img)
    # Background subtle garden gradient
    for y in range(400):
        c = int(240 - y * 0.05)
        draw.line([(0, y), (400, y)], fill=(c, c+10, c))
    
    # Leaf outline
    leaf_poly = [(200, 30), (270, 90), (330, 180), (310, 290), (240, 350), (200, 370), (160, 350), (90, 290), (70, 180), (130, 90)]
    draw.polygon(leaf_poly, fill=main_color)
    
    # Primary & Secondary leaf veins
    draw.line([(200, 30), (200, 370)], fill=(20, 80, 30), width=4)
    for vy in range(80, 340, 35):
        draw.line([(200, vy), (280, vy-25)], fill=(30, 95, 40), width=2)
        draw.line([(200, vy), (120, vy-25)], fill=(30, 95, 40), width=2)
    
    # Petiole stem
    draw.line([(200, 370), (200, 395)], fill=(15, 60, 20), width=6)
    
    # Draw necrotic lesions / spots if disease present
    if spots > 0 and spot_color:
        np.random.seed(42)
        for _ in range(spots):
            sx = int(np.random.normal(200, 50))
            sy = int(np.random.normal(200, 60))
            r = int(np.random.randint(6, 22))
            if 100 < sx < 300 and 60 < sy < 330:
                draw.ellipse([sx-r, sy-r, sx+r, sy+r], fill=spot_color, outline=(40, 20, 10))
                if blight:
                    # Concentric rings of Alternaria early blight
                    draw.ellipse([sx-r//2, sy-r//2, sx+r//2, sy+r//2], fill=(45, 25, 12))
    
    if rust:
        np.random.seed(101)
        for _ in range(40):
            rx = int(np.random.randint(120, 280))
            ry = int(np.random.randint(80, 320))
            draw.ellipse([rx-3, ry-4, rx+3, ry+4], fill=(210, 95, 15))
            
    img = img.filter(ImageFilter.SMOOTH)
    img.save(os.path.join("static", "samples", filename), quality=92)
    print(f"Created sample image: static/samples/{filename}")

def create_sample_tree(filename, canopy_color, bark_color=(100, 65, 35)):
    img = Image.new("RGB", (400, 400), (235, 245, 252))
    draw = ImageDraw.Draw(img)
    # Sky background
    for y in range(300):
        b = int(245 - y * 0.15)
        draw.line([(0, y), (400, y)], fill=(215, 235, b))
    # Ground soil
    draw.rectangle([0, 300, 400, 400], fill=(70, 120, 60))
    
    # Trunk
    draw.polygon([(180, 360), (220, 360), (210, 190), (190, 190)], fill=bark_color)
    draw.line([(190, 220), (140, 170)], fill=bark_color, width=8)
    draw.line([(210, 210), (260, 160)], fill=bark_color, width=8)
    
    # Canopy foliage
    draw.ellipse([90, 50, 310, 220], fill=canopy_color)
    draw.ellipse([60, 90, 220, 230], fill=canopy_color)
    draw.ellipse([180, 80, 340, 230], fill=canopy_color)
    draw.ellipse([120, 40, 280, 160], fill=(canopy_color[0]+15, canopy_color[1]+20, canopy_color[2]+10))
    
    img = img.filter(ImageFilter.SMOOTH)
    img.save(os.path.join("static", "samples", filename), quality=92)
    print(f"Created tree sample: static/samples/{filename}")

# Generate sample crop leaf images
create_sample_leaf("sample_tomato_early_blight.jpg", (95, 155, 60), spot_color=(90, 50, 25), spots=15, blight=True)
create_sample_leaf("sample_potato_late_blight.jpg", (80, 135, 55), spot_color=(50, 35, 20), spots=18, blight=True)
create_sample_leaf("sample_apple_scab.jpg", (100, 165, 70), spot_color=(55, 50, 30), spots=22)
create_sample_leaf("sample_corn_rust.jpg", (120, 175, 75), rust=True)
create_sample_leaf("sample_rice_blast.jpg", (105, 170, 65), spot_color=(120, 60, 20), spots=12)
create_sample_leaf("sample_cotton_blight.jpg", (90, 150, 60), spot_color=(70, 40, 20), spots=16)
create_sample_leaf("sample_healthy_leaf.jpg", (45, 165, 65), spots=0)

# Generate sample tree images
create_sample_tree("sample_tree_neem.jpg", (35, 140, 65))
create_sample_tree("sample_tree_teak.jpg", (50, 125, 45), bark_color=(115, 75, 40))
create_sample_tree("sample_tree_banyan.jpg", (25, 115, 45), bark_color=(95, 80, 65))
create_sample_tree("sample_tree_mango.jpg", (40, 150, 50))
create_sample_tree("sample_tree_sandalwood.jpg", (60, 160, 75), bark_color=(80, 55, 35))

print("All labels, models, and sample images successfully generated!")
