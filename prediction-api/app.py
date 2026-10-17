import os

from flask import Flask, request, jsonify

from house_price_predictor import HousePricePredictor

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/house_price_predictor', methods=['POST'])
def predict_str():
    prediction_input = request.get_json()
    price = hpp.predict_single_record(prediction_input)
    return jsonify({'result': price}), 200


hpp = HousePricePredictor()
if __name__ == '__main__':
    app.run(port=int(os.getenv("PORT", 5000)), host='0.0.0.0', debug=True)