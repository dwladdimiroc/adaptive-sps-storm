import numpy as np
from sklearn import svm

def svm_prediction(samples, prediction_number):
    t = list(range(len(samples)))

    regression_svm = svm.SVR().fit(np.array(t).reshape(-1, 1), samples)

    predictions = []
    for x in range(prediction_number):
        predictions.append(regression_svm.predict(np.array([len(samples) + x]).reshape(-1, 1))[0])
        # print(f"predicted f({len(samples) + x}): {mlp.predict(np.array([len(samples) + x]).reshape(-1, 1))}")

    next_input = np.average(predictions)
    return next_input, predictions
