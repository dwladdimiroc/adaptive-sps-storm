package storm

import (
	"encoding/json"
	"fmt"
	"github.com/spf13/viper"
	"io"
	"net/http"
	"strings"
	"time"
)

const NimbusSummaryTopologiesBaseURL = "http://UI_HOST:UI_PORT/api/v1/topology/summary"
const NimbusSummaryTopologyBaseURL = "http://UI_HOST:UI_PORT/api/v1/topology/TOPOLOGY_ID"
const NimbusComponentsBaseURL = "http://UI_HOST:UI_PORT/api/v1/topology/TOPOLOGY_ID/component/COMPONENT_ID"

func parseURL(urlRaw string, topologyId string) string {
	var url string
	nimbusHost := viper.GetString("nimbus.host")
	nimbusPort := viper.GetString("nimbus.port")
	url = strings.Replace(urlRaw, "UI_HOST", nimbusHost, 1)
	url = strings.Replace(url, "UI_PORT", nimbusPort, 1)
	url = strings.Replace(url, "TOPOLOGY_ID", topologyId, 1)
	return url
}

func parseComponentURL(urlRaw string, topologyId string, component string) string {
	var url string
	nimbusHost := viper.GetString("nimbus.host")
	nimbusPort := viper.GetString("nimbus.port")
	url = strings.Replace(urlRaw, "UI_HOST", nimbusHost, 1)
	url = strings.Replace(url, "UI_PORT", nimbusPort, 1)
	url = strings.Replace(url, "TOPOLOGY_ID", topologyId, 1)
	url = strings.Replace(url, "COMPONENT_ID", component, 1)
	return url
}

func GetTopologyId() string {
	var summaryTopologies SummaryTopologies

	nimbusSummaryTopologies := parseURL(NimbusSummaryTopologiesBaseURL, "")
	if res, err := http.Get(nimbusSummaryTopologies); err != nil {
		fmt.Printf("storm get summary topologies: %v\n", err)
	} else {
		data, _ := io.ReadAll(res.Body)
		if err := res.Body.Close(); err != nil {
			fmt.Printf("storm get summary topologies: %v\n", err)
		} else {
			if err := json.Unmarshal(data, &summaryTopologies); err != nil {
				fmt.Printf("storm get summary topologies: %v\n", err)
			}
		}
	}

	if len(summaryTopologies.Topologies) > 0 {
		return summaryTopologies.Topologies[0].Id
	} else {
		time.Sleep(1 * time.Second)
		return GetTopologyId()
	}
}

func GetSummaryTopology(topologyId string) SummaryTopology {
	var summaryTopology SummaryTopology

	nimbusSummaryTopologyURL := parseURL(NimbusSummaryTopologyBaseURL, topologyId)
	if res, err := http.Get(nimbusSummaryTopologyURL); err != nil {
		fmt.Printf("storm get summary topology: %v\n", err)
	} else {
		data, _ := io.ReadAll(res.Body)
		if err := res.Body.Close(); err != nil {
			fmt.Printf("storm get summary topology: %v\n", err)
		} else {
			if err := json.Unmarshal(data, &summaryTopology); err != nil {
				fmt.Printf("storm get summary topology: %v\n", err)
			}
		}
	}

	if len(summaryTopology.Bolts) > 0 {
		return summaryTopology
	} else {
		time.Sleep(1 * time.Second)
		return GetSummaryTopology(topologyId)
	}
}

func GetMetrics(topology Topology) (bool, TopologyMetrics) {
	var metricsTopology TopologyMetrics
	for _, spout := range topology.Spouts {
		metricsTopology.Spouts = append(metricsTopology.Spouts, GetComponentSpout(topology.Id, spout.Name))
	}
	for _, bolt := range topology.Bolts {
		metricsTopology.Bolts = append(metricsTopology.Bolts, GetComponentBolt(topology.Id, bolt.Name))
	}
	return true, metricsTopology
}

func GetComponentBolt(topologyId, boltName string) BoltMetrics {
	var boltMetrics BoltMetrics

	nimbusSummaryTopologyURL := parseComponentURL(NimbusComponentsBaseURL, topologyId, boltName)
	if res, err := http.Get(nimbusSummaryTopologyURL); err != nil {
		fmt.Printf("storm get summary topology: %v\n", err)
	} else {
		data, _ := io.ReadAll(res.Body)
		if err := res.Body.Close(); err != nil {
			fmt.Printf("storm get summary topology: %v\n", err)
		} else {
			if err := json.Unmarshal(data, &boltMetrics); err != nil {
				fmt.Printf("storm get summary topology: %v\n", err)
			}
		}
	}

	return boltMetrics
}

func GetComponentSpout(topologyId, spoutName string) SpoutMetrics {
	var spoutMetrics SpoutMetrics

	nimbusSummaryTopologyURL := parseComponentURL(NimbusComponentsBaseURL, topologyId, spoutName)
	if res, err := http.Get(nimbusSummaryTopologyURL); err != nil {
		fmt.Printf("storm get summary topology: %v\n", err)
	} else {
		data, _ := io.ReadAll(res.Body)
		if err := res.Body.Close(); err != nil {
			fmt.Printf("storm get summary topology: %v\n", err)
		} else {
			if err := json.Unmarshal(data, &spoutMetrics); err != nil {
				fmt.Printf("storm get summary topology: %v\n", err)
			}
		}
	}

	return spoutMetrics
}
