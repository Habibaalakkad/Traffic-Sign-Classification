

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# ==============================================
# PARAMETERS
# ==============================================
IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CHANNELS = 3  # RGB

# ⚠️ Change these paths
TRAIN_CSV = r"F:\elevvo\traffic\Train.csv"
TEST_CSV  = r"F:\elevvo\traffic\Test.csv"
META_FILE = r"F:\elevvo\traffic\Meta.csv"
DATA_DIR  = r"F:\elevvo\traffic"   # root folder where images are stored

# ==============================================
# 1. LOAD META DATA (CLASS NAMES)
# ==============================================
meta_data = pd.read_csv(META_FILE)

# Clean column names
meta_data.columns = meta_data.columns.str.strip().str.replace(";", "")

print("[INFO] Meta.csv columns:", meta_data.columns.tolist())

# Build mapping if columns exist
if "ClassId" in meta_data.columns and "SignName" in meta_data.columns:
    class_id_to_name = dict(zip(meta_data["ClassId"], meta_data["SignName"]))
else:
    print("[WARNING] 'ClassId' or 'SignName' not found in Meta.csv!")
    # fallback: just use numeric labels
    class_id_to_name = {}

# ==============================================
# 2. LOAD DATASET FROM CSV
# ==============================================
def load_data_from_csv(csv_file, data_dir):
    data = pd.read_csv(csv_file)
    images, labels = [], []

    for _, row in data.iterrows():
        img_path = os.path.join(data_dir, row["Path"])  # 'Path' column contains relative paths
        label = int(row["ClassId"])
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            labels.append(label)
        except Exception as e:
            print("[ERROR] Failed to load:", img_path, "->", e)
            continue

    return np.array(images), np.array(labels)

print("[INFO] Loading training dataset...")
X_train, y_train = load_data_from_csv(TRAIN_CSV, DATA_DIR)

print("[INFO] Loading testing dataset...")
X_test, y_test = load_data_from_csv(TEST_CSV, DATA_DIR)

print("Train set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)

num_classes = len(np.unique(y_train))

# ==============================================
# 3. PREPROCESS DATA
# ==============================================
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

y_train = to_categorical(y_train, num_classes=num_classes)
y_test  = to_categorical(y_test, num_classes=num_classes)

# ==============================================
# 4. BUILD CNN MODEL
# ==============================================
def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

model = build_model(num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==============================================
# 5. TRAIN MODEL
# ==============================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=64,
    verbose=1
)

# ==============================================
# 6. EVALUATION
# ==============================================
print("[INFO] Evaluating model...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification Report
if class_id_to_name:
    target_names = [class_id_to_name.get(i, str(i)) for i in range(num_classes)]
else:
    target_names = [str(i) for i in range(num_classes)]

print(classification_report(y_true, y_pred_classes, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(20, 15))
sns.heatmap(cm, annot=False, cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ==============================================
# 7. PLOT TRAINING RESULTS
# ==============================================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.legend()
plt.title("Loss")

plt.show()


