import joblib
import pickle
import numpy as np
import json

from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils
import tensorflow as tf

class CustomPredictor(Predictor):
    def __init__(self):
        return
    
    def load(self, artifacts_uri: str):
        fs = Featurestore(
           featurestore_name="ccc_promotions",
           project="datafusionsbox",
           location="us-central1",
        )

        self._stats_entity_type = fs.get_entity_type(entity_type_id="stats")
        self._model = tf.keras.models.load_model('model_artifacts')
    
    ##def preprocess(self, prediction_input: Any):
        
    def predict(self, instances):
        user_ids = instances["instances"]
        instances = self._stats_entity_type.read(entity_ids=[user_ids]) ##get all features , feature_ids=_features)
        inputs = instances.values.tolist()[:][1:]
        outputs = self._model.predict([inputs])
        return outputs.tolist()
        
    def postprocess(self, prediction_results: list):
        return { "predictions": prediction_results }
    