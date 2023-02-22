import joblib
import pickle
import numpy as np
import json

from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils
import tensorflow as tf
from google.cloud import aiplatform
from google.cloud.aiplatform import Featurestore
import numpy as np

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
        print("user ids",user_ids)
        instance_ids=list(np.concatenate(user_ids))
        instances = self._stats_entity_type.read(entity_ids=instance_ids) ##get all features , feature_ids=_features)
        inputs = instances.drop("entity_id", axis=1)
        pinputs = inputs.values.tolist()

        outputs = self._model.predict(pinputs)
        print(outputs)
        return outputs.tolist()
        
    def postprocess(self, prediction_results: list):
        return { "predictions": prediction_results }
    