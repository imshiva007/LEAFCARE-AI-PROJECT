from pathlib import Path
import tensorflow as tf

# ==============================
# 1. Project paths
# ==============================

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "plantdoc_data" / "train"
TEST_DIR = BASE_DIR / "plantdoc_data" / "test"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 5
SEED = 42

print("Loading PlantDoc dataset...")

# ==============================
# 2. Load training dataset
# ==============================

train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    validation_split=0.2,
    subset="training",
    seed=SEED
)

# ==============================
# 3. Load validation dataset
# ==============================

val_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    validation_split=0.2,
    subset="validation",
    seed=SEED
)

# Save the exact class order learned by the model
class_names = train_data.class_names
number_of_classes = len(class_names)

print("\nNumber of classes:", number_of_classes)
print("Classes:", class_names)

# ==============================
# 4. Load test dataset
#    using training class order
# ==============================

test_class_names = [
    name for name in class_names
    if (TEST_DIR / name).is_dir()
]

missing_test_classes = [
    name for name in class_names
    if name not in test_class_names
]

print("\nClasses missing from test dataset:")
print(missing_test_classes)

if not test_class_names:
    raise ValueError("Test dataset mein matching class folders nahi mile.")

test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=test_class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# Map test labels to the original training class indices
test_label_map = {
    name: index for index, name in enumerate(class_names)
}

test_label_indices = [
    test_label_map[name] for name in test_class_names
]

test_data = test_data.map(
    lambda images, labels: (
        images,
        tf.gather(
            tf.constant(test_label_indices, dtype=tf.int32),
            labels
        )
    )
)

# ==============================
# 5. Improve dataset performance
# ==============================

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)
test_data = test_data.prefetch(AUTOTUNE)

# ==============================
# 6. Build CNN model
# ==============================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(160, 160, 3)),

    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.Conv2D(
        32, 3, activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64, 3, activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        128, 3, activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        128, activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        number_of_classes,
        activation="softmax"
    )
])

# ==============================
# 7. Compile model
# ==============================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# 8. Train model
# ==============================

print("\nStarting model training...")

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ==============================
# 9. Evaluate model
# ==============================

print("\nEvaluating on test dataset...")

test_loss, test_accuracy = model.evaluate(test_data)

print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")

# ==============================
# 10. Save trained model
# ==============================

model_path = MODEL_DIR / "leaf_disease_model.keras"

model.save(model_path)

# Save class labels in exact training order
class_names_path = MODEL_DIR / "class_names.txt"

with open(
    class_names_path,
    "w",
    encoding="utf-8"
) as file:
    file.write("\n".join(class_names))

print("\nTraining complete!")
print("Model saved at:", model_path)
print("Class names saved at:", class_names_path)