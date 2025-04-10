package adaptive

import (
	"github.com/dwladdimiroc/sps-storm/internal/storm"
	"github.com/spf13/viper"
	"log"
)

func planning(topology *storm.Topology) {
	for i := range topology.Bolts {
		if topology.Bolts[i].PredictionReplicas < 1 {
			topology.Bolts[i].Replicas = 1
		} else {
			if topology.Bolts[i].PredictionReplicas > viper.GetInt64("storm.adaptive.limit_replicas") {
				topology.Bolts[i].Replicas = viper.GetInt64("storm.adaptive.limit_replicas")
			} else {
				topology.Bolts[i].Replicas = topology.Bolts[i].PredictionReplicas
			}
		}
		log.Printf("planning: ok\n")
		log.Printf("planning: bolt={%s},replicas={%d}\n", topology.Bolts[i].Name, topology.Bolts[i].Replicas)
	}
	execute(*topology)
}
