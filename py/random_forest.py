import numpy as np

from sklearn.ensemble import RandomForestRegressor


def rf_prediction(samples, prediction_number):
    # for i in range(len(samples)):
        # samples[i] = samples[i] / 1000

    # data = pd.DataFrame(samples, columns=['input'])
    #
    # forecaster = ForecasterAutoreg(
    #     regressor=RandomForestRegressor(random_state=123),
    #     lags=6
    # )
    #
    # forecaster.fit(y=data['input'])
    # data = forecaster.predict(prediction_number)
    # print(data)
    # predictions = []
    # for i in range(len(data.values())):
    #     predictions.append(data.values()[i][0])
    #
    # next_input = np.average(predictions)
    # return next_input, predictions

    t = list(range(len(samples)))
    # print(t)
    # print(samples)
    rand_forest = RandomForestRegressor()
    rand_forest.fit(np.array(t).reshape(-1, 1), samples)

    predictions = []
    for x in range(prediction_number):
        predictions.append(rand_forest.predict(np.array([len(samples) + x]).reshape(-1, 1))[0])
        # print(f"predicted f({len(samples) + x}): {rand_forest.predict(np.array([len(samples) + x]).reshape(-1, 1))}")

    next_input = np.average(predictions)
    return next_input, predictions
