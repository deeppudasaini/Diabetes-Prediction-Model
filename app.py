from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('svm.pkl', 'rb'))


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()


@app.route("/predict", methods=['POST'])
def predict():

    if request.method == 'POST':
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        DiabetesPedigreeFunction = float(
            request.form['DiabetesPedigreeFunction'])
        SkinThickness = int(request.form['SkinThickness'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        Age = int(request.form['Age'])

        prediction = model.predict(
            [[Pregnancies, Glucose, BloodPressure, DiabetesPedigreeFunction, SkinThickness, Insulin, BMI, Age]])
        print(prediction)
        if prediction[0] == 0:
            return render_template('index.html', prediction_texts="The patient does not have diabetes")
        elif prediction[0] == 1:
            return render_template('index.html', prediction_texts="The patient is diagnosed with diabetes")

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
