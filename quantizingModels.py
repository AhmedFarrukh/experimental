import tensorflow as tf
import pathlib

modelNames = ["MobileNet", "InceptionV3", "ResNet50", "ResNet101" , "ResNet152", "VGG16", "VGG19"]

for modelName in modelNames:
  model_class = getattr(tf.keras.applications, modelName)
  model = model_class(weights='imagenet')

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model_quant = converter.convert()

  tflite_models_dir = pathlib.Path("./tflite_models/")
  tflite_models_dir.mkdir(exist_ok=True, parents=True)

  # Save the unquantized/float model:
  tflite_model_file = tflite_models_dir/(modelName+".tflite")
  tflite_model_file.write_bytes(tflite_model)
  # Save the quantized model:
  tflite_model_quant_file = tflite_models_dir/(modelName+"_quant.tflite")
  tflite_model_quant_file.write_bytes(tflite_model_quant)