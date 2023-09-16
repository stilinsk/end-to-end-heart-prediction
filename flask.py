import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
regmodel = pickle.load(open('trained_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    input_data = np.array(list(data.values())).reshape(1, -1)
    output = regmodel.predict(input_data)
    if output[0] == 0:
        prediction_text = 'The patient is healthy'
    else:
        prediction_text ='The patient has a heart disease'
    return jsonify(prediction_text)

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    input_data = np.array(data).reshape(1, -1)
    output = regmodel.predict(input_data)[0]
    if output == 0:
        prediction_text ='The patient is healthy'
    else:
        prediction_text = 'The patient has a heart disease'
    return render_template("index.html", prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)