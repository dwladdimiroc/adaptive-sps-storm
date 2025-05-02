import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def gaussian_prediction(samples, prediction_number):
    t = list(range(len(samples)))

    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9).fit(np.array(t).reshape(-1, 1), samples)

    predictions = []
    for x in range(prediction_number):
        predictions.append(gaussian_process.predict(np.array([len(samples) + x]).reshape(-1, 1))[0])
        # print(f"predicted f({len(samples) + x}): {mlp.predict(np.array([len(samples) + x]).reshape(-1, 1))}")

    next_input = np.average(predictions)
    return next_input, predictions
