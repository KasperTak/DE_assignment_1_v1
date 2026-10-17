import json
import os
import logging
import pickle
from io import StringIO

import pandas as pd

CATEGORIES = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]


class HousePricePredictor:
    def __init__(self):
        self.model = None

    def predict_single_record(self, prediction_input):
        logging.debug(prediction_input)
        if self.model is None:
            try:
                # laad het model uit de map in de environment variable MODEL_REPO
                model_repo = os.environ['MODEL_REPO']
                file_path = os.path.join(model_repo, "model.pkl")
                self.load_model(file_path)
            except KeyError:
                print("MODEL_REPO is undefined")
                # anders het lokale model (in de prediction-api map)
                self.load_model('model.pkl')

        # accepteert zowel een dict als een lijst met één dict
        if isinstance(prediction_input, dict):
            prediction_input = [prediction_input]
        df = pd.read_json(StringIO(json.dumps(prediction_input)), orient='records')

        # one-hot encoding van ocean_proximity, zoals in split_data
        for cat in CATEGORIES:
            df[f"ocean_proximity_{cat}"] = (df["ocean_proximity"] == cat).astype(int)

        # precies de kolommen (en volgorde) waarop het model is getraind
        xNew = df[list(self.model.feature_names_in_)]
        y_pred = self.model.predict(xNew)
        logging.info(y_pred[0])
        return float(y_pred[0])

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)