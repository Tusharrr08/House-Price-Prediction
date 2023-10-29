from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('models/regressor.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = request.form['bedrooms']
    val2 = request.form['bathrooms']
    val3 = request.form['stories']
    val4 = request.form['mainroad']
    val5 = request.form['guestroom']
    val6 = request.form['basement']
    val7 = request.form['hotwaterheating']
    val8 = request.form['airconditioning']
    val9 = request.form['parking']
    val10 = request.form['prefarea']
    val11 = request.form['Furnished']
    val12 = request.form['Semi-Furnished']
    arr = np.array([val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])

    return render_template('index.html', data=int(pred))


if __name__ == '__main__':
    app.run(debug=True)
