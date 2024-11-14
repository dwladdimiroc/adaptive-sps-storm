package predictive

import (
	"github.com/dwladdimiroc/sps-storm/internal/storm"
	"github.com/spf13/viper"
	"log"
	"math"
	"math/rand"
	"sync"
)

var indexChosenPredictor int
var predictions []PredictionInput

type PredictionInput struct {
	NameModel       string
	PredictedInput  []float64
	ErrorEstimation float64
}

func GetAllPred() []PredictionInput {
	return predictions
}

func GetPred() PredictionInput {
	return predictions[indexChosenPredictor]
}

func InitPrediction() {
	var basic, lr, fft, ann, rf, svm, bayesian, ridge, sgd PredictionInput
	//[0] Basic
	basic.NameModel = "basic"
	basic.PredictedInput = make([]float64, viper.GetInt("storm.adaptive.analyze_samples"))
	predictions = append(predictions, basic)
	//[1] LR
	lr.NameModel = "linear_regression"
	lr.PredictedInput = make([]float64, viper.GetInt("storm.adaptive.analyze_samples"))
	predictions = append(predictions, lr)
	//[2] FFT
	fft.NameModel = "fft"
	fft.PredictedInput = make([]float64, viper.GetInt("storm.adaptive.analyze_samples"))
	predictions = append(predictions, fft)
	//[3] RF
	rf.NameModel = "random_forest"
	rf.PredictedInput = make([]float64, viper.GetInt("storm.adaptive.analyze_samples"))
	predictions = append(predictions, rf)
	//[4] MPL (ANN)
	ann.NameModel = "ann"
	ann.PredictedInput = make([]float64, viper.GetInt("storm.adaptive.analyze_samples"))
	predictions = append(predictions, ann)
	//[5] SVM
	svm.NameModel = "svm"
	svm.PredictedInput = make([]float64, viper.GetInt("storm.adaptive.analyze_samples"))
	predictions = append(predictions, svm)
	//[6] Bayesian
	bayesian.NameModel = "bayesian"
	bayesian.PredictedInput = make([]float64, viper.GetInt("storm.adaptive.analyze_samples"))
	predictions = append(predictions, bayesian)
	//[7] Ridge
	ridge.NameModel = "ridge"
	ridge.PredictedInput = make([]float64, viper.GetInt("storm.adaptive.analyze_samples"))
	predictions = append(predictions, ridge)
	//Gaussian
	//gaussian.NameModel = "gaussian"
	//gaussian.PredictedInput = make([]float64, viper.GetInt("storm.adaptive.analyze_samples"))
	//predictions = append(predictions, gaussian)
	//[8] SGD
	sgd.NameModel = "sgd"
	sgd.PredictedInput = make([]float64, viper.GetInt("storm.adaptive.analyze_samples"))
	predictions = append(predictions, sgd)

	if viper.GetString("storm.adaptive.predictive_model") != "multi" {
		for i := 0; i < len(predictions); i++ {
			if predictions[i].NameModel == viper.GetString("storm.adaptive.predictive_model") {
				var singlePrediction = predictions[i]
				predictions = nil
				predictions = append(predictions, singlePrediction)
			}
		}
	}
}

func PredictInput(topology *storm.Topology) {
	if len(predictions) == 1 {
		if viper.GetString("storm.adaptive.predictive_model") == "basic" {
			predictions[0].PredictedInput = append(predictions[0].PredictedInput, Simple(topology)...)
		} else {
			var wg sync.WaitGroup
			wg.Add(1)
			go GetPredictionInput(topology, 0, &wg)
			wg.Wait()
		}
	} else {
		predictions[0].PredictedInput = append(predictions[0].PredictedInput, Simple(topology)...)
		var wg sync.WaitGroup
		for i := 1; i < len(predictions); i++ {
			wg.Add(1)
			go GetPredictionInput(topology, i, &wg)
		}

		wg.Wait()
	}
}

func GetPredictionInput(topology *storm.Topology, indexPredictor int, wg *sync.WaitGroup) {
	defer wg.Done()

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
	resultsPrediction := GetPrediction(samples, viper.GetInt("storm.adaptive.prediction_number"), predictions[indexPredictor].NameModel)
	if len(resultsPrediction) > 0 {
		predictions[indexPredictor].PredictedInput = append(predictions[indexPredictor].PredictedInput, resultsPrediction...)
	}
}

//TODO Refactor this function

func DeterminatePredictor(topology *storm.Topology) {
	if len(predictions) == 1 {
		log.Printf("Not used selector model")
		indexChosenPredictor = 0
	} else {
		var selectorModel = viper.GetString("storm.adaptive.selector_model")
		if selectorModel == "rmse" {
			//Calculate error
			calculateError(topology.InputRate)

			//Determinate the best predictor
			indexPredictor := 0
			minErrorEstimation := predictions[0].ErrorEstimation
			//log.Printf("[t=X] model={%s},RMSE={%.2f}", predictions[0].NameModel, predictions[0].ErrorEstimation)
			for i := 1; i < len(predictions); i++ {
				//log.Printf("[t=X] model={%s},RMSE={%.2f}", predictions[i].NameModel, predictions[i].ErrorEstimation)
				if predictions[i].ErrorEstimation < minErrorEstimation {
					indexPredictor = i
					minErrorEstimation = predictions[i].ErrorEstimation
				}
			}
			indexChosenPredictor = indexPredictor
		} else if selectorModel == "bandit-greedy" {
			value := rand.Float64()
			if viper.GetString("storm.deploy.dataset") == "sbac" {
				// GREEDY - TWITTER
				if 0.0 < value && value <= 0.687 {
					indexChosenPredictor = 4 //ANN
				} else if 0.687 < value && value <= 0.932 {
					indexChosenPredictor = 6 //Bayesian
				} else { //0.932 < value && value <= 1
					indexChosenPredictor = 8 //SGD
				}
			} else if viper.GetString("storm.deploy.dataset") == "dns" {
				// GREEDY - DNS
				if 0.0 < value && value <= 0.615 {
					indexChosenPredictor = 4 //ANN
				} else if 0.615 < value && value <= 0.687 {
					indexChosenPredictor = 6 //Bayesian
				} else { //0.687 < value && value <= 1
					indexChosenPredictor = 8 //SGD
				}
			} else {
				indexChosenPredictor = 0
			}
			log.Printf("[t=X] selector_model={%s},prediction_model={%s},rand_value={%.3f}", selectorModel, predictions[indexChosenPredictor].NameModel, value)
		} else if selectorModel == "bandit-ucb" {
			value := rand.Float64()
			if viper.GetString("storm.deploy.dataset") == "sbac" {
				// UCB - TWITTER
				if 0.0 < value && value <= 0.687 {
					indexChosenPredictor = 4 //ANN
				} else if 0.687 < value && value <= 0.932 {
					indexChosenPredictor = 6 //Bayesian
				} else { //0.932 < value && value <= 1
					indexChosenPredictor = 8 //SGD
				}
			} else if viper.GetString("storm.deploy.dataset") == "dns" {
				// UCB - DNS
				if 0.0 < value && value <= 0.167 {
					indexChosenPredictor = 4 //ANN
				} else if 0.167 < value && value <= 0.296 {
					indexChosenPredictor = 6 //Bayesian
				} else { //0.296 < value && value <= 1
					indexChosenPredictor = 8 //SGD
				}
			} else {
				indexChosenPredictor = 0
			}
			log.Printf("[t=X] selector_model={%s},prediction_model={%s},rand_value={%.3f}", selectorModel, predictions[indexChosenPredictor].NameModel, value)
		} else if selectorModel == "bandit-greedy-avg" {
			value := rand.Float64()
			// GREEDY - AVG
			if 0.0 < value && value <= 0.651 {
				indexChosenPredictor = 4 //ANN
			} else if 0.651 < value && value <= 0.79 {
				indexChosenPredictor = 6 //Bayesian
			} else { //0.79 < value && value <= 1
				indexChosenPredictor = 8 //SGD
			}
			log.Printf("[t=X] selector_model={%s},prediction_model={%s},rand_value={%.3f}", selectorModel, predictions[indexChosenPredictor].NameModel, value)
		} else if selectorModel == "bandit-ucb-avg" {
			value := rand.Float64()
			// UCB - AVG
			if 0.0 < value && value <= 0.276 {
				indexChosenPredictor = 4 //ANN
			} else if 0.276 < value && value <= 0.709 {
				indexChosenPredictor = 6 //Bayesian
			} else { //0.709 < value && value <= 1
				indexChosenPredictor = 8 //SGD
			}
			log.Printf("[t=X] selector_model={%s},prediction_model={%s},rand_value={%.3f}", selectorModel, predictions[indexChosenPredictor].NameModel, value)
		} else {
			indexChosenPredictor = 0
		}
	}
}

// RMSE
func calculateError(input []int64) {
	var timeWindow = viper.GetInt("storm.adaptive.analyze_samples") + viper.GetInt("storm.adaptive.prediction_number")
	for p := 0; p < len(predictions); p++ {
		var errorEst float64
		//log.Printf("[t=X] model={%s},lenPrediction={%d},lenInput={%d}", predictions[p].NameModel, len(predictions[p].PredictedInput), len(input))
		for i := len(input) - timeWindow; i < len(input); i++ {
			errorEst += math.Pow(predictions[p].PredictedInput[i]-float64(input[i]), 2.0)
		}
		predictions[p].ErrorEstimation = math.Sqrt(errorEst / float64(timeWindow))
	}
}

func GetPredictedInputPeriod(period int) int64 {
	if period >= len(predictions[indexChosenPredictor].PredictedInput) {
		period = len(predictions[indexChosenPredictor].PredictedInput) - 1
	}
	predictedInputPeriod := int64(predictions[indexChosenPredictor].PredictedInput[period])
	//log.Printf("predicted input period : %d perdiction={%v}", period, predictions[indexChosenPredictor])
	return predictedInputPeriod
}
