package adaptive

import (
	"github.com/dwladdimiroc/sps-storm/internal/predictive"
	"github.com/dwladdimiroc/sps-storm/internal/storm"
	"github.com/dwladdimiroc/sps-storm/internal/util"
	"github.com/jasonlvhit/gocron"
	"github.com/spf13/viper"
	"log"
	"time"
)

var topology *storm.Topology
var period int
var schedulerAdaptive *gocron.Scheduler

func Init(topologyId string) {
	topology = new(storm.Topology)
	topology.Init(topologyId)
	summaryTopology := storm.GetSummaryTopology(topology.Id)
	topology.CreateTopology(summaryTopology)
	topology.InitReplicas()
	log.Printf("Topology created\n")
	go util.InitServer()
	predictive.InitPrediction()
	schedulerAdaptive = gocron.NewScheduler()
}

func Start(limit time.Duration) {
	go func(schedulerAdaptive *gocron.Scheduler) {
		if err := schedulerAdaptive.Every(uint64(viper.GetInt("storm.adaptive.time_window_size"))).Seconds().Do(adaptiveSystem, topology); err != nil {
			log.Printf("scheduler: fatal error={%v}", err)
			return
		}
		<-schedulerAdaptive.Start()
	}(schedulerAdaptive)
	time.Sleep(limit)
}

func adaptiveSystem(topology *storm.Topology) {
	if ok := monitor(topology); ok {
		if viper.GetBool("storm.deploy.analyze") {
			analyze(topology)
		}
	}
	topology.ClearStatsTimeWindow()
}

func Stop() {
	schedulerAdaptive.Clear()
}
