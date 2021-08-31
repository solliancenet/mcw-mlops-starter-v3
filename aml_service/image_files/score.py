import os
import json
import numpy as np
import pandas as pd

import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from azureml.core.model import Model
from azureml.monitoring import ModelDataCollector
import onnxruntime

def init():
    global model
    global inputs_dc, prediction_dc
    global tokenizer, max_len, max_words
    
    try:
        model_name = 'compliance-classifier'
        print('Looking for model path for model: ', model_name)
        model_path = Model.get_model_path(model_name = model_name)
        print('Loading model from: ', model_path)
        # Load the ONNX model
        model = onnxruntime.InferenceSession(model_path)
        print('Model loaded...')

        inputs_dc = ModelDataCollector("model_telemetry", designation="inputs")
        prediction_dc = ModelDataCollector("model_telemetry", designation="predictions", feature_names=["prediction"])

        cardata_url = ('https://quickstartsws9073123377.blob.core.windows.net/'
                        'azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/'
                        'quickstarts/connected-car-data/connected-car_components.csv')

        car_components_descriptions = pd.read_csv(cardata_url)['text'].tolist()
        print('Training dataset loaded...')

        max_len = 100
        max_words = 10000
        tokenizer = Tokenizer(num_words = max_words)
        tokenizer.fit_on_texts(car_components_descriptions)
        print('Tokenizer fitted...')

    except Exception as e:
        print(e)

# note you can pass in multiple rows for scoring
def run(raw_data):
    import time
    try:
        print("Received input: ", raw_data)
        
        inputs = json.loads(raw_data)     

        sequences = tokenizer.texts_to_sequences(inputs)
        data = pad_sequences(sequences, max_len, dtype=np.float32)

        results = model.run(None, {model.get_inputs()[0].name:data})[0]
        results = results.flatten()

        inputs_dc.collect(inputs) #this call is saving our input data into Azure Blob
        prediction_dc.collect(results) #this call is saving our output data into Azure Blob

        print("Prediction created " + time.strftime("%H:%M:%S"))
        
        return json.dumps(results.tolist())
    except Exception as e:
        error = str(e)
        print("ERROR: " + error + " " + time.strftime("%H:%M:%S"))
        return error
