import numpy as np
from PIL import Image
import tensorflow as tf
import json

# Load class indices
with open('utils/class_indices.json' , 'r') as f:
    class_indices = json.load(f)
    
# Reverse mapping: class name -> index
index_to_class = {int(v): k for v , k in class_indices.items()}


def load_and_preprocess_image(image , target_size = (224,224)):
    img = Image.open(image).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img , dtype=np.float32)
    img_array = img_array / 255.0                   # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
    return img_array

def predict_disease_from_model(image , model_path="model/plant_disease_model_quant.tflite"):
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_array = load_and_preprocess_image(image)
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred_index = np.argmax(output)
    print(pred_index)
    return pred_index
    # pred_class = index_to_class[pred_index]
    # return pred_class

