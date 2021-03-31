# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 21:16:25 2021

@author: Karthik C V
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app_name = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app_name.route("/")
def home():
    return render_template("index.html")

@app_name.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Car Price is {}'.format(output))


if __name__ == "__main__":
    app_name.run(debug=True)