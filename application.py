import json
import pickle

from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

application = Flask(__name__)

app = application

# Load the model and preocessor
mexico_model = pickle.load(open('mexico_model.pkl','rb'))
mexico_processor = pickle.load(open('mexico_processor.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Define features
    features = [
                 'property_type',
                 'borough',
                 'surface_covered_in_m2',
                 'price_per_m2',
                 'lat',
                 'lon'
                ]

    # Retrive values from form
    values = [x for x in request.form.values()]
    
    # Define dataframe
    data = pd.DataFrame(dict(list(zip(features, values))), index=[0])

    # perform prediction
    prediction = mexico_model.predict(mexico_processor.transform(pd.DataFrame(data, index=[0])))[0]

    return render_template('home.html', prediction_text='The Home Value Is: {}'.format(prediction))


if __name__=="__main__":
    app.run(debug=True)