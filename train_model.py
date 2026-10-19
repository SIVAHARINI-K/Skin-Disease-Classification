import os
import numpy as np
import pandas as pd
import cv2
import pickle
from glob import glob
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Dense,
    GlobalAveragePooling2D
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =====================================================
# CONFIG
# =====================================================

IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 7

DATASET_DIR = "dataset"
MODEL_PATH = "skin_disease_model.h5"
LABEL_PATH = "label_encoder.pkl"

np.random.seed(42)
tf.random.set_seed(42)

# =====================================================
# FIND IMAGE
# =====================================================

def find_image(image_id, dirs):
    for d in dirs:
        for ext in [".jpg", ".jpeg", ".png"]:
            path = os.path.join(d, image_id + ext)
            if os.path.exists(path):
                return path
    return None

# =====================================================
# LOAD DATA
# =====================================================

def load_data():

    csv_files = glob(os.path.join(DATASET_DIR, "**/*.csv"), recursive=True)

    if len(csv_files) == 0:
        raise FileNotFoundError("HAM10000 metadata CSV not found")

    csv_file = csv_files[0]

    df = pd.read_csv(csv_file)

    print("\n========== CLASS DISTRIBUTION ==========")
    print(df["dx"].value_counts())
    print("========================================\n")

    image_dirs = [
        os.path.join(DATASET_DIR, "HAM10000_images_part_1"),
        os.path.join(DATASET_DIR, "HAM10000_images_part_2"),
        DATASET_DIR
    ]

    images = []
    labels = []

    print("[INFO] Loading Images...")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        img_path = find_image(row["image_id"], image_dirs)

        if img_path is None:
            continue

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        images.append(img)
        labels.append(row["dx"])

    images = np.array(images, dtype=np.float32) / 255.0

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    print("\n[INFO] Classes:")
    print(list(le.classes_))

    return images, labels, le

# =====================================================
# BUILD MODEL
# =====================================================

def build_model():

    model = Sequential()

    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(
        Conv2D(
            64,
            (3, 3),
            activation="relu"
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.30))

    model.add(
        Conv2D(
            128,
            (3, 3),
            activation="relu"
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.35))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.50))

    model.add(Dense(NUM_CLASSES, activation="softmax"))

    return model

# =====================================================
# MAIN
# =====================================================

def main():

    print("[INFO] Loading Dataset...")

    X, y, le = load_data()

    with open(LABEL_PATH, "wb") as f:
        pickle.dump(le, f)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        stratify=y_train,
        random_state=42
    )

    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)

    classes = np.unique(y_train)

    cw = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = {
        i: cw[i] for i in range(len(cw))
    }

    print("\n[INFO] Class Weights:")
    print(class_weights)

    train_gen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=False
    )

    val_gen = ImageDataGenerator()

    train_data = train_gen.flow(
        X_train,
        y_train_cat,
        batch_size=BATCH_SIZE
    )

    val_data = val_gen.flow(
        X_val,
        y_val_cat,
        batch_size=BATCH_SIZE
    )

    model = build_model()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    steps_per_epoch = int(np.ceil(len(X_train) / BATCH_SIZE))
    val_steps = int(np.ceil(len(X_val) / BATCH_SIZE))

    print("\n[INFO] Training Started...\n")

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        class_weight=class_weights,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    print("\n[INFO] Evaluating Model...\n")

    y_pred = np.argmax(
        model.predict(X_test),
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\n================================")
    print("TEST ACCURACY:", accuracy)
    print("================================\n")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    model.save(MODEL_PATH)

    print("\n[SUCCESS] Model Saved Successfully")
    print("[SUCCESS] Saved:", MODEL_PATH)
    print("[SUCCESS] Saved:", LABEL_PATH)

if __name__ == "__main__":
    main()