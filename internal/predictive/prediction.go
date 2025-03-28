package predictive

import (
	"github.com/dwladdimiroc/sps-storm/internal/storm"
	"github.com/spf13/viper"
)

var predictions PredictionInput

type PredictionInput struct {
	NameModel      string
	PredictedInput []float64
}

func GetPred() PredictionInput {
	return predictions
}

func InitPrediction() {
	predictions.NameModel = viper.GetString("storm.adaptive.predictive_model")
	predictions.PredictedInput = make([]float64, viper.GetInt("storm.adaptive.analyze_samples"))
}

func PredictInput(topology *storm.Topology) {
	var samples []float64

	var index int
	if index = len(topology.InputRate) - viper.GetInt("storm.adaptive.prediction_samples"); index < 0 {
		index = 0
	}
	for i := index; i < len(topology.InputRate); i++ {
		samples = append(samples, float64(topology.InputRate[i]))
		//log.Printf("analyze: train: index={%d},sample={%v},\n", i, topology.InputRate[i])
	}

	//log.Printf("[t=X] predict input : init prediction")
	resultsPrediction := GetPrediction(samples, viper.GetInt("storm.adaptive.prediction_number"), predictions.NameModel)
	if len(resultsPrediction) > 0 {
		predictions.PredictedInput = append(predictions.PredictedInput, resultsPrediction...)
	}
}

func GetPredictedInputPeriod(period int) int64 {
	if period >= len(predictions.PredictedInput) {
		period = len(predictions.PredictedInput) - 1
	}
	predictedInputPeriod := int64(predictions.PredictedInput[period])
	//log.Printf("predicted input period : %d perdiction={%v}", period, predictions[indexChosenPredictor])
	return predictedInputPeriod
}
