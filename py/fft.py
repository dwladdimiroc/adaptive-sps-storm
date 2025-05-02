import numpy as np

from darts import TimeSeries
from darts.models import FFT


def fft_prediction(samples, prediction_number):
    timeseries = TimeSeries.from_values(np.array(samples).reshape(-1, 1))
    fft = FFT(trend="poly")
    fft.fit(timeseries)

    #print(fft)
    #print(timeseries)

    data = fft.predict(prediction_number)
    #print(data)
    predictions = []
    for i in range(len(data.values())):
        predictions.append(data.values()[i][0])

    next_input = np.average(predictions)
    return next_input, predictions
