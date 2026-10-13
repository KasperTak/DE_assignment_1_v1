import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  accuracy_score, recall_score
import logging
import os
from pathlib import Path
from flask import jsonify
import pickle

def train(dataset):
    # split into input (X) and output (y) variables
    X = dataset.drop("median_house_value", axis=1)
    y = dataset["median_house_value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    val_predictions = lr_model.predict(X_test)
    accuracy = accuracy_score(y_test, val_predictions)
    recall = recall_score(y_test, val_predictions)
    print(f"Accuracy: {accuracy:.3f} | Recall:{recall:.3f}")
    #evaluate the model
    text_out = {
        "accuracy:": accuracy,
        "recall:": recall,
        }
    logging.info(text_out)
    
    #Saving model in a given location provided as an env. variable
    model_repo = os.getenv("MODEL_REPO")
    if model_repo:
        Path(model_repo).mkdir(parents=True, exist_ok=True) # create if it does not exist
        file_path = os.path.join(model_repo, "model.pkl")
        # save object to pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(lr_model, f)
        logging.info("Saved the model to the location: " + model_repo)
        return jsonify(text_out), 200
    else:
        # otherwise, save the model locally
        with open("model.pkl", 'wb') as f:
            pickle.dump(lr_model, f)
        return jsonify({'message': 'The model was saved locally.'}), 200

