import numpy as np
from sklearn import linear_model


def ridge_prediction(samples, prediction_number):
    t = list(range(len(samples)))

    ridge = linear_model.Ridge().fit(np.array(t).reshape(-1, 1), samples)

    predictions = []
    for x in range(prediction_number):
        predictions.append(ridge.predict(np.array([len(samples) + x]).reshape(-1, 1))[0])
        # print(f"predicted f({len(samples) + x}): {mlp.predict(np.array([len(samples) + x]).reshape(-1, 1))}")

    next_input = np.average(predictions)
    return next_input, predictions
