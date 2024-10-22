from flask import Flask, request, jsonify

from ann import ann_prediction
from bayesian import bayesian_prediction
from fft import fft_prediction
from gaussian import gaussian_prediction
from linear_regression import linear_regression_prediction
from random_forest import rf_prediction
from ridge import ridge_prediction
from sgd import sgd_prediction
from svm import svm_prediction

app = Flask(__name__)


@app.route('/')
def test():  # put application's code here
    return 'Test'


@app.route('/random_forest', methods=["POST"])
def random_forest():
    samples = request.json['samples']
    prediction_number = request.json['prediction_number']
    avg_prediction, predictions = rf_prediction(samples, prediction_number)
    resp = {
        "avg_prediction": avg_prediction,
        "predictions": predictions,
    }
    return jsonify(resp)


@app.route('/ann', methods=["POST"])
def ann():
    samples = request.json['samples']
    prediction_number = request.json['prediction_number']
    avg_prediction, predictions = ann_prediction(samples, prediction_number)
    resp = {
        "avg_prediction": avg_prediction,
        "predictions": predictions,
    }
    return jsonify(resp)


@app.route('/fft', methods=["POST"])
def fft():
    samples = request.json['samples']
    prediction_number = request.json['prediction_number']
    avg_prediction, predictions = fft_prediction(samples, prediction_number)
    resp = {
        "avg_prediction": avg_prediction,
        "predictions": predictions,
    }
    return jsonify(resp)


@app.route('/linear_regression', methods=["POST"])
def linear_regression():
    samples = request.json['samples']
    prediction_number = request.json['prediction_number']
    avg_prediction, predictions = linear_regression_prediction(samples, prediction_number)
    resp = {
        "avg_prediction": avg_prediction,
        "predictions": predictions,
    }
    return jsonify(resp)

@app.route('/svm', methods=["POST"])
def svm():
    samples = request.json['samples']
    prediction_number = request.json['prediction_number']
    avg_prediction, predictions = svm_prediction(samples, prediction_number)
    resp = {
        "avg_prediction": avg_prediction,
        "predictions": predictions,
    }
    return jsonify(resp)

@app.route('/bayesian', methods=["POST"])
def bayesian():
    samples = request.json['samples']
    prediction_number = request.json['prediction_number']
    avg_prediction, predictions = bayesian_prediction(samples, prediction_number)
    resp = {
        "avg_prediction": avg_prediction,
        "predictions": predictions,
    }
    return jsonify(resp)

@app.route('/ridge', methods=["POST"])
def ridge():
    samples = request.json['samples']
    prediction_number = request.json['prediction_number']
    avg_prediction, predictions = ridge_prediction(samples, prediction_number)
    resp = {
        "avg_prediction": avg_prediction,
        "predictions": predictions,
    }
    return jsonify(resp)

@app.route('/gaussian', methods=["POST"])
def gaussian():
    samples = request.json['samples']
    prediction_number = request.json['prediction_number']
    avg_prediction, predictions = gaussian_prediction(samples, prediction_number)
    resp = {
        "avg_prediction": avg_prediction,
        "predictions": predictions,
    }
    return jsonify(resp)

@app.route('/sgd', methods=["POST"])
def sgd():
    samples = request.json['samples']
    prediction_number = request.json['prediction_number']
    avg_prediction, predictions = sgd_prediction(samples, prediction_number)
    resp = {
        "avg_prediction": avg_prediction,
        "predictions": predictions,
    }
    return jsonify(resp)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
