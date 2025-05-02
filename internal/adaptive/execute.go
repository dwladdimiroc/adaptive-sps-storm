package adaptive

import (
	"github.com/dwladdimiroc/sps-storm/internal/storm"
	"github.com/dwladdimiroc/sps-storm/internal/util"
	"log"
	"strconv"
)

func execute(topology storm.Topology) {
	if err := updateReplicas(topology); err != nil {
		log.Printf("execute: rebalanced topology {%v}\n", err)
	}
	//else {
	//	log.Printf("execute: rebalanced topology {ok}\n")
	//}
}

func updateReplicas(topology storm.Topology) error {
	var err error
	for _, bolt := range topology.Bolts {
		value := strconv.FormatInt(bolt.Replicas, 10)
		if errRedis := util.RedisSet(bolt.Name, value); errRedis != nil {
			log.Printf("update replicas error: %v\n", errRedis)
			err = errRedis
		}
	}

	return err
}
