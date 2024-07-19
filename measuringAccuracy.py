import tensorflow as tf
import os
import tensorflow_datasets as tfds
import numpy as np

ds_name = 'imagenet2012'
ds, info = tfds.load(ds_name, split='validation', as_supervised=True, with_info=True)

def inference(model_path, modelName):
    tflite_model_path = model_path
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    correct_predictions = 0
    total_predictions = 0

    preProcessDetails = {
        "MobileNet": ([224, 224], "mobilenet"),
        "ResNet50": ([224, 224], "resnet"),
        "ResNet101": ([224, 224], "resnet"),
        "InceptionV3": ([299, 299], "inception_v3"),
        "VGG16": ([224, 224], "vgg16"),
        "VGG19": ([224, 224], "vgg19"),
        "ResNet152": ([224, 224], "resnet")
    }

    def preprocess(image, label):
        image = tf.image.resize(image, preProcessDetails[modelName][0])
        model = getattr(tf.keras.applications, preProcessDetails[modelName][1])
        image = model.preprocess_input(image)
        return image, label

    # Batch processing and prefetching
    batch_size = 32  # Adjust the batch size depending on your system's capabilities
    ds_processed = ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    for images, labels in ds_processed:
        for i in range(images.shape[0]):
            single_image = np.expand_dims(images.numpy()[i], axis=0)
            single_label = labels.numpy()[i]
            interpreter.set_tensor(input_index, single_image)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_index)
            if np.argmax(predictions[0]) == single_label:
                correct_predictions += 1
            total_predictions += 1

    return correct_predictions / total_predictions

modelNames = ["MobileNet"]
for modelName in modelNames:
    accuracy = inference("./tflite_models/" + modelName + ".tflite", modelName)
    print(f"{modelName}: {accuracy:.2%}")
    #quant_accuracy = inference("./tflite_models/" + modelName + "_quant.tflite", modelName)
    #print(f"{modelName} Quantized: {quant_accuracy:.2%}")

