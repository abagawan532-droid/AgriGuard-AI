"""Train a basic transfer-learning leaf classifier from an ImageFolder dataset.
Dataset layout: datasets/leaf/train/<class_name>/*.jpg and datasets/leaf/validation/<class_name>/*.jpg
Use a dataset you have permission to use; review class labels and licenses first.
"""
import os, json
import tensorflow as tf
from tensorflow.keras import layers, models

ROOT = os.path.join("datasets", "leaf")
TRAIN = os.path.join(ROOT, "train")
VAL = os.path.join(ROOT, "validation")
OUT = "models"
os.makedirs(OUT, exist_ok=True)
if not os.path.isdir(TRAIN) or not os.path.isdir(VAL):
    raise SystemExit("Create datasets/leaf/train and datasets/leaf/validation folders first. See BUILD_GUIDE.md.")

train = tf.keras.utils.image_dataset_from_directory(TRAIN, image_size=(224,224), batch_size=32)
val = tf.keras.utils.image_dataset_from_directory(VAL, image_size=(224,224), batch_size=32, shuffle=False)
names = train.class_names
with open("data/leaf_labels.json","w",encoding="utf-8") as f:
    json.dump(names, f, ensure_ascii=False, indent=2)
base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights="imagenet")
base.trainable = False
inputs = layers.Input((224,224,3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(len(names), activation="softmax")(x)
model = models.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=8)
model.save(os.path.join(OUT,"leaf_model.keras"))
print("Saved models/leaf_model.keras. Evaluate on independent images before use.")
