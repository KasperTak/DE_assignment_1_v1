import json
import os

import house_price_predictor


class TestHousePricePredictor:
    def test_predict_single_record(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        test_data_file = os.path.join(test_dir, "../testResources/prediction_request.json")
        test_model_file = os.path.join(test_dir, "../testResources/dtr_dtr_model.pkl")
        with open(test_data_file) as json_file:
            data = json.load(json_file)
        hpp = house_price_predictor.HousePricePredictor()
        hpp.load_model(test_model_file)
        price = hpp.predict_single_record(data)
        assert price is not None
        assert price > 0
