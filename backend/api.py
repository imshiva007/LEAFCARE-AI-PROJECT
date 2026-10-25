from pathlib import Path
from io import BytesIO

import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Use the improved model
MODEL_PATH = (
    BASE_DIR / "models" / "leaf_disease_model_improved.keras"
)

CLASS_NAMES_PATH = (
    BASE_DIR / "models" / "class_names_improved.txt"
)

# Improved model was trained with 224 x 224 images
IMAGE_SIZE = (224, 224)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="LeafCare AI API",
    description="Educational plant leaf disease prediction API",
    version="2.0.0",
)


# Allow local frontend development servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ==========================================
# MODEL VARIABLES
# ==========================================

model = None
class_names = []


# ==========================================
# LOAD MODEL WHEN SERVER STARTS
# ==========================================

@app.on_event("startup")
def load_model():
    global model, class_names

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Improved model not found: {MODEL_PATH}"
        )

    if not CLASS_NAMES_PATH.exists():
        raise RuntimeError(
            f"Improved class names file not found: {CLASS_NAMES_PATH}"
        )

    # Load improved trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load class names in their original training order
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = [
            line.strip()
            for line in file
            if line.strip()
        ]

    output_count = int(model.output_shape[-1])

    if len(class_names) != output_count:
        model = None
        raise RuntimeError(
            f"Model has {output_count} outputs, but "
            f"class names file has {len(class_names)} classes."
        )

    print("\n" + "=" * 45)
    print("LEAFCARE AI — IMPROVED MODEL")
    print("=" * 45)
    print("Model loaded successfully!")
    print(f"Model file: {MODEL_PATH.name}")
    print(f"Input size: {IMAGE_SIZE}")
    print(f"Number of classes: {len(class_names)}")
    print("=" * 45 + "\n")


# ==========================================
# HOME / HEALTH CHECK
# ==========================================

@app.get("/")
def home():
    return {
        "message": "LeafCare AI API is running",
        "model_loaded": model is not None,
        "model_name": MODEL_PATH.name,
        "number_of_classes": len(class_names),
        "image_size": IMAGE_SIZE,
    }


# ==========================================
# GET AVAILABLE CLASSES
# ==========================================

@app.get("/classes")
def get_classes():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="AI model is not loaded.",
        )

    return {
        "success": True,
        "total_classes": len(class_names),
        "classes": class_names,
    }


# ==========================================
# PREDICT LEAF DISEASE
# ==========================================

@app.post("/predict")
async def predict_leaf(file: UploadFile = File(...)):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="AI model is not loaded.",
        )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, PNG, or WEBP image.",
        )

    try:
        # Read uploaded image
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded image is empty.",
            )

        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image must be smaller than 10 MB.",
            )

        # Open and prepare image
        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        # Resize to match improved model input
        image = image.resize(IMAGE_SIZE)

        # Match MobileNetV2 training preprocessing:
        # pixel values [0, 255] -> [-1, 1]
        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

        image_array = image_array / 127.5 - 1.0

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        # Run prediction
        predictions = model.predict(
            image_array,
            verbose=0,
        )[0]

        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_class = class_names[predicted_index]

        confidence = float(
            predictions[predicted_index]
        ) * 100

        # Return the top three model predictions
        top_indices = np.argsort(predictions)[-3:][::-1]

        top_predictions = [
            {
                "class_name": class_names[int(index)],
                "confidence": round(
                    float(predictions[index]) * 100,
                    2,
                ),
            }
            for index in top_indices
        ]

        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "top_predictions": top_predictions,
            "disclaimer": (
                "Educational prototype only. Predictions may be "
                "incorrect and are not a confirmed plant diagnosis."
            ),
        }

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    except HTTPException:
        raise

    except Exception as error:
        print(f"Prediction error: {error}")

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed. Check the backend terminal "
                "for details."
            ),
        )

    finally:
        await file.close()