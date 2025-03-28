package adaptive

import (
	"github.com/dwladdimiroc/sps-storm/internal/predictive"
	"github.com/dwladdimiroc/sps-storm/internal/storm"
	"github.com/dwladdimiroc/sps-storm/internal/util"
	"github.com/spf13/viper"
	"log"
	"math"
)

func analyze(topology *storm.Topology) {
	//log.Printf("analyze: period %v\n", period)
	if period%viper.GetInt("storm.adaptive.analyze_samples") == 0 {
		log.Printf("[t=%d] analyze: prediction\n", period)
		// Safe prediction, if the prediction is not ready before the next analyze
		simplesPrediction := predictive.Simple(topology)
		//log.Printf("[t=%d] analyze: predictedInput={%d},simplesPrediction={%d}", period, len(topology.PredictedInputRate), len(simplesPrediction))
		for i := 0; i < len(simplesPrediction); i++ {
			topology.PredictedInputRate = append(topology.PredictedInputRate, int64(simplesPrediction[i]))
		}

		predictive.PredictInput(topology)
		for i := range topology.Bolts {
			topology.Bolts[i].PredictionQueue = predictionInputQueue(topology.Bolts[i], *topology) / viper.GetInt64("storm.adaptive.analyze_samples")
		}

		topology.ClearQueue()

		//log.Printf("[t=%d] analyze: predictedModel={%s},predictedInput={%d},topologyInput={%d}", period, predictive.GetPred().NameModel, len(predictive.GetPred().PredictedInput), len(topology.PredictedInputRate))
		init := len(predictive.GetPred().PredictedInput) - viper.GetInt("storm.adaptive.prediction_number")
		for i := init; i < len(predictive.GetPred().PredictedInput); i++ {
			topology.PredictedInputRate[i] = int64(predictive.GetPred().PredictedInput[i])
		}
	}

	//log.Printf("input predicted: %d\n", input)
	if period >= viper.GetInt("storm.adaptive.analyze_samples") && period%viper.GetInt("storm.adaptive.planning_samples") == 0 {
		log.Printf("[t=%d] analyze: determinate replicas\n", period)
		for i := range topology.Bolts {
			var predictedInput int64
			for j := 0; j < viper.GetInt("storm.adaptive.planning_samples"); j++ {
				predictedInput += predictive.GetPredictedInputPeriod(period + j)
			}
			predictedInput /= viper.GetInt64("storm.adaptive.planning_samples")
			predictedInput += topology.Bolts[i].PredictionQueue
			topology.Bolts[i].PredictionReplicas = predictionReplicas(predictedInput, topology.Bolts[i])
			//log.Printf("[t=%d] analyze: bolt={%s},predictionInput={%d},predictionReplicas={%d}", period, topology.Bolts[i].Name, predictedInput, topology.Bolts[i].PredictionReplicas)
		}
		planning(topology)
	}
}

func predictionInputQueue(bolt storm.Bolt, topology storm.Topology) int64 {
	var valuePredictionQ = bolt.Queue
	for _, tagBoltPredecessor := range bolt.BoltsPredecessor {
		for _, boltTopology := range topology.Bolts {
			if boltTopology.Name == tagBoltPredecessor {
				valuePredictionQ += predictionInputQueue(boltTopology, topology)
			}
		}
	}

	return valuePredictionQ
}

func predictionReplicas(input int64, bolt storm.Bolt) int64 {
	executedTimeAvg := chooseExecutedTime(bolt)
	timeWindow := float64(int64(viper.GetInt("storm.adaptive.time_window_size")) * util.SECS)
	replicasPredictive := float64(input) * executedTimeAvg / timeWindow
	//log.Printf("analyze: prediction replicas={%v},input={%v},execTime={%v},timeWindow={%v}\n", replicasPredictive, input, executedTimeAvg, timeWindow)
	return int64(math.Ceil(replicasPredictive))
}

func chooseExecutedTime(bolt storm.Bolt) float64 {
	executedTimeAvg := bolt.GetExecutedTimeAvg()
	if bolt.ExecutedTimeBenchmarkAvg > executedTimeAvg {
		return bolt.ExecutedTimeBenchmarkAvg
	} else {
		return executedTimeAvg
	}
}
