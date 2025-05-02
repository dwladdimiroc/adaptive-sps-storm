import numpy as np

from sklearn.linear_model import LinearRegression


def linear_regression_prediction(samples, prediction_number):
    t = list(range(len(samples)))
    # print(t)
    # print(samples)
    regression_linear = LinearRegression().fit(np.array(t).reshape(-1, 1), samples)

    predictions = []
    for x in range(prediction_number):
        predictions.append(regression_linear.predict(np.array([len(samples) + x]).reshape(-1, 1))[0])

    next_input = np.average(predictions)
    return next_input, predictions
