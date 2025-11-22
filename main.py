from fastapi import FastAPI, HTTPException, UploadFile
from starlette.middleware.cors import CORSMiddleware
from ai_edge_litert.interpreter import Interpreter
from PIL import Image
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/cnn_lite.tflite"
interpreter = Interpreter(MODEL_PATH)
interpreter.allocate_tensors()

# Ambil input dan output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize(IMG_SIZE)

        # Preprocess
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        predictions = interpreter.get_tensor(output_details[0]['index'])
        score = predictions[0]

        return {
            "filename": file.filename,
            "predicted_class": CLASS_NAMES[np.argmax(score)],
            "confidence": float(100 * np.max(score)),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something wrong : {str(e)}")
