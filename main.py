from fastapi import FastAPI, HTTPException, UploadFile
import tensorflow as tf
import numpy as np
import keras
import io

app = FastAPI()

MODEL_PATH = "models/cnn.keras"
model = keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Bacterial_Spot",
    "Early_Blight",
    "Healthy",
    "Late_Blight",
    "Leaf_Mold",
    "Mosaic_Virus",
    "Septoria_Leaf_Spot",
    "Spider_Mites",
    "Target_Spot",
    "Yellow_Leaf_Curl_Virus",
]

IMG_SIZE = (224, 224)


@app.get("/", response_model=dict)
def read_root() -> dict:
    return {
        "project": "Tomato Disease Detection",
        "version": "1.0",
        "info": "Check the /docs routes",
    }


@app.post("/predict")
async def predict(file: UploadFile):
    if file.content_type != "image/jpeg":
        raise HTTPException(
            422, "The uploaded file must be a image file (.jpeg or .jpg)."
        )

    try:
        # Read File
        contents = await file.read()
        img = keras.utils.load_img(io.BytesIO(contents), target_size=IMG_SIZE)

        # Preprocess
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # Prediction
        predictions = model.predict(img_array)
        score = predictions[0]

        return {
            "filename": file.filename,
            "predicted_class": CLASS_NAMES[np.argmax(score)],
            "confidence": float(100 * np.max(score)),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something wrong : {str(e)}")
