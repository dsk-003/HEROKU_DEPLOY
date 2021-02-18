# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:33:53 2021

@author: DHANSHREE
"""

import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html',prediction_text='Prediction Price of Diamond Appr is RS{}'.format(output))

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering result on HTML GUI
    
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 4)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True) 