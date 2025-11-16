import tensorflow as tf

MODEL_PATH = "models/cnn/cnn_final.keras"
converter = tf.lite.TFLiteConverter.from_keras_model(MODEL_PATH)
tflite_model = converter.convert()

with open('models/cnn_lite.tflite', 'wb') as file:
    file.write(tflite_model)