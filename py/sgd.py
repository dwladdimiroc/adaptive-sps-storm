import numpy as np
from sklearn.linear_model import SGDRegressor


def sgd_prediction(samples, prediction_number):
    t = list(range(len(samples)))

    sgd_regressor = SGDRegressor().fit(np.array(t).reshape(-1, 1), samples)

    predictions = []
    for x in range(prediction_number):
        predictions.append(sgd_regressor.predict(np.array([len(samples) + x]).reshape(-1, 1))[0])
        # print(f"predicted f({len(samples) + x}): {mlp.predict(np.array([len(samples) + x]).reshape(-1, 1))}")

    next_input = np.average(predictions)
    return next_input, predictions
