package predictive

import (
	"github.com/dwladdimiroc/sps-storm/internal/storm"
	"github.com/spf13/viper"
	"log"
)

func Simple(topology *storm.Topology) []float64 {
	var predictionBasic []float64

	var index int
	if index = len(topology.InputRate) - viper.GetInt("storm.adaptive.prediction_samples"); index < 0 {
		index = 0
	}

	for i := index; i < len(topology.InputRate); i++ {
		log.Printf("simple prediction : %v\n", topology.InputRate[i])
		predictionBasic = append(predictionBasic, float64(topology.InputRate[i]))
	}
	return predictionBasic
}
