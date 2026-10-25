from pathlib import Path
import tensorflow as tf

# Project folders
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "plantdoc_data" / "train"
TEST_DIR = BASE_DIR / "plantdoc_data" / "test"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 5
SEED = 42

print("Loading dataset...")

# Load training and validation datasets
train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    label_mode="int"
)

val_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    label_mode="int"
)

class_names = train_data.class_names
num_classes = len(class_names)

print("Number of classes:", num_classes)

# Load only test classes that exist in training
test_class_names = [
    name for name in class_names
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

# Convert test labels to training class indices
label_map = {name: i for i, name in enumerate(class_names)}
test_indices = [label_map[name] for name in test_class_names]

test_data = test_data.map(
    lambda images, labels: (
        images,
        tf.gather(tf.constant(test_indices), labels)
    )
)

# Improve input pipeline performance
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)
test_data = test_data.prefetch(AUTOTUNE)

# Download pretrained MobileNetV2 weights
print("Loading pretrained MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# Build transfer learning model
inputs = tf.keras.Input(shape=(160, 160, 3))

# MobileNetV2 expects pixels scaled to [-1, 1]
x = tf.keras.layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(inputs)

x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train classifier head
print("Starting transfer learning...")

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Evaluate on available test classes
print("Evaluating model...")
test_loss, test_accuracy = model.evaluate(test_data)

print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Save model and class labels
model.save(MODEL_DIR / "leaf_disease_model.keras")

with open(
    MODEL_DIR / "class_names.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("\n".join(class_names))

print("Training complete!")
print("Model saved:", MODEL_DIR / "leaf_disease_model.keras")