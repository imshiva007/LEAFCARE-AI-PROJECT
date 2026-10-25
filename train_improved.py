from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_DIR = BASE_DIR / "plantdoc_data" / "train"
TEST_DIR = BASE_DIR / "plantdoc_data" / "test"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "leaf_disease_model_improved.keras"
CLASSES_PATH = MODEL_DIR / "class_names_improved.txt"
REPORT_PATH = MODEL_DIR / "evaluation_report.txt"
HISTORY_PATH = MODEL_DIR / "training_history.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
HEAD_EPOCHS = 15
FINE_TUNE_EPOCHS = 10
SEED = 42

tf.keras.utils.set_random_seed(SEED)

print("=" * 55)
print("LEAFCARE AI — IMPROVED MODEL TRAINING")
print("=" * 55)


# ==========================================
# 2. CHECK DATASET
# ==========================================

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training directory not found: {TRAIN_DIR}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found: {TEST_DIR}"
    )


# ==========================================
# 3. LOAD TRAINING AND VALIDATION DATA
# ==========================================

print("\nLoading training and validation datasets...")

train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    label_mode="int",
    shuffle=True
)

class_names = train_data.class_names
num_classes = len(class_names)

val_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    label_mode="int",
    shuffle=False
)

print(f"\nNumber of classes: {num_classes}")

for index, name in enumerate(class_names):
    print(f"{index}: {name}")


# ==========================================
# 4. COMPUTE CLASS WEIGHTS
# ==========================================

print("\nCalculating class weights...")

train_labels = np.concatenate([
    labels.numpy()
    for _, labels in train_data
])

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=train_labels
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(class_weights_array)
}

# Limit extreme weights so classes with very few images
# do not dominate training.
class_weights = {
    index: min(weight, 5.0)
    for index, weight in class_weights.items()
}

print("Class weights calculated.")


# ==========================================
# 5. DATA AUGMENTATION
# ==========================================

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomContrast(0.10),
], name="data_augmentation")


# ==========================================
# 6. OPTIMIZE DATA PIPELINE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)


# ==========================================
# 7. LOAD MOBILENETV2
# ==========================================

print("\nLoading pretrained MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


# ==========================================
# 8. BUILD MODEL
# ==========================================

inputs = tf.keras.Input(
    shape=(*IMAGE_SIZE, 3),
    name="leaf_image"
)

x = augmentation(inputs)

x = tf.keras.layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(x)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.35)(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(x)

x = tf.keras.layers.Dropout(0.25)(x)

outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax",
    name="predictions"
)(x)

model = tf.keras.Model(inputs, outputs)

model.summary()


# ==========================================
# 9. CALLBACKS
# ==========================================

def make_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]


# ==========================================
# 10. TRAIN CLASSIFICATION HEAD
# ==========================================

print("\n" + "=" * 55)
print("PHASE 1: TRAINING CLASSIFICATION HEAD")
print("=" * 55)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_head = model.fit(
    train_data,
    validation_data=val_data,
    epochs=HEAD_EPOCHS,
    class_weight=class_weights,
    callbacks=make_callbacks()
)


# ==========================================
# 11. FINE-TUNE MOBILENETV2
# ==========================================

print("\n" + "=" * 55)
print("PHASE 2: FINE-TUNING MOBILENETV2")
print("=" * 55)

base_model.trainable = True

# Freeze most of the pretrained layers.
# Fine-tune only the later layers.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Batch normalization is kept frozen for stability.
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=make_callbacks()
)


# ==========================================
# 12. SAVE TRAINING HISTORY
# ==========================================

combined_history = {}

for key in history_head.history:
    combined_history[key] = (
        history_head.history[key]
        + history_fine.history.get(key, [])
    )

with open(HISTORY_PATH, "w", encoding="utf-8") as file:
    json.dump(combined_history, file, indent=2)

print("\nTraining history saved:", HISTORY_PATH)


# ==========================================
# 13. SAVE MODEL AND CLASS LABELS
# ==========================================

print("\nSaving improved model...")

model.save(MODEL_PATH)

with open(CLASSES_PATH, "w", encoding="utf-8") as file:
    file.write("\n".join(class_names))

print("Model saved:", MODEL_PATH)
print("Class names saved:", CLASSES_PATH)


# ==========================================
# 14. LOAD TEST DATASET
# ==========================================

print("\nLoading test dataset...")

# Keep only test classes represented in training.
test_class_names = [
    name
    for name in class_names
    if (TEST_DIR / name).is_dir()
]

test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=test_class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# Map test labels to the corresponding training class indices.
label_map = {
    name: index
    for index, name in enumerate(class_names)
}

test_indices = [
    label_map[name]
    for name in test_class_names
]

test_data = test_data.map(
    lambda images, labels: (
        images,
        tf.gather(
            tf.constant(test_indices, dtype=tf.int32),
            labels
        )
    )
).prefetch(AUTOTUNE)


# ==========================================
# 15. EVALUATE MODEL
# ==========================================

print("\n" + "=" * 55)
print("FINAL TEST EVALUATION")
print("=" * 55)

test_loss, test_accuracy = model.evaluate(
    test_data,
    verbose=1
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ==========================================
# 16. CLASSIFICATION REPORT
# ==========================================

print("\nGenerating classification report...")

true_labels = []
predicted_labels = []

for images, labels in test_data:
    probabilities = model.predict(
        images,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    true_labels.extend(labels.numpy().tolist())
    predicted_labels.extend(predictions.tolist())


report = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(num_classes)),
    target_names=class_names,
    zero_division=0
)

print("\n", report)

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write(
        f"Test Loss: {test_loss:.4f}\n"
        f"Test Accuracy: {test_accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "Note: Classes absent from the test set have no test examples. "
        "Their reported metrics are not evidence of performance.\n\n"
    )

    file.write(report)

print("\nEvaluation report saved:", REPORT_PATH)


# ==========================================
# 17. CONFUSION MATRIX
# ==========================================

matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=list(range(num_classes))
)

print("\nConfusion matrix shape:", matrix.shape)

print("\n" + "=" * 55)
print("TRAINING AND EVALUATION COMPLETED")
print("=" * 55)

print("\nNew model:", MODEL_PATH)
print("Old model has not been overwritten.")
print("Do not connect the new model to the website until")
print("its performance has been reviewed.")