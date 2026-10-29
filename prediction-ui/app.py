# importing Flask and other modules
import json
import os
import logging
import requests
from flask import Flask, request, render_template, jsonify

# Flask constructor
app = Flask(__name__)

# A decorator used to tell the application
# which URL is associated function
@app.route('/predicthouseprice', methods=["GET", "POST"])

def predict_house_price():
    if request.method == "GET":
        return render_template("input_form_page.html")

    elif request.method == "POST":
        prediction_input = [ #fields from input page
            {
                "housing_median_age": int(request.form.get("mah")),
                "total_rooms": int(request.form.get("trb")),
                "total_bedrooms": int(request.form.get("tbb")),
                "population": int(request.form.get("tprb")),
                "households": int(request.form.get("thb")),
                "median_income": float(request.form.get("mihb")),
                "ocean_proximity": request.form.get("op")
            }
        ]

        app.logger.debug("Prediction input : %s", prediction_input)

        # use requests library to execute the prediction service API by sending an HTTP POST request
        # use an environment variable to find the value of the house price prediction API
        # json.dumps() function will convert a subset of Python objects into a json string.
        # json.loads() method can be used to parse a valid JSON string and convert it into a Python Dictionary.
        predictor_api_url = os.environ['PREDICTOR_API']
        res = requests.post(predictor_api_url, json=json.loads(json.dumps(prediction_input)))

        prediction_value = res.json()['result']
        app.logger.info("Prediction Output : %s", prediction_value)
        return render_template("output_form_page.html",
                               prediction_variable=round(float(prediction_value), 2)) 

    else:
        return jsonify(message="Method Not Allowed"), 405  # The 405 Method Not Allowed should be used to indicate
    # that our app that does not allow the users to perform any other HTTP method (e.g., PUT and  DELETE) for
    # '/predicthouseprice' path


# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
