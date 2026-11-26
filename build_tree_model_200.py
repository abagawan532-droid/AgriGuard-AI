import json
import tensorflow as tf
from tensorflow.keras import layers, models

def build_tree_model():
    with open("data/tree_labels.json", "r", encoding="utf-8") as f:
        tree_labels = json.load(f)

    print(f"Building models/tree_model.keras with {len(tree_labels)} classes...")
    inputs_tree = layers.Input(shape=(224, 224, 3), name="tree_input")
    x = layers.Rescaling(1.0 / 255)(inputs_tree)
    base_tree = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    x = base_tree(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs_tree = layers.Dense(len(tree_labels), activation="softmax", name="tree_predictions")(x)
    model_tree = models.Model(inputs_tree, outputs_tree, name="AgriGuard_Tree_Classifier_200")
    model_tree.save("models/tree_model.keras")
    print(f"[OK] Saved models/tree_model.keras with {len(tree_labels)} output units successfully!")

if __name__ == "__main__":
    build_tree_model()
