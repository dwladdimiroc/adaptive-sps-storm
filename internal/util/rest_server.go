package util

import (
	"encoding/json"
	"github.com/spf13/viper"
	"log"
	"net/http"
)

var latency float64

type RequestData struct {
	Latency float64 `json:"latency"`
}

func InitServer() {
	http.HandleFunc("/sendLatency", sendLatency)
	log.Println("server: init")
	http.ListenAndServe(":"+viper.GetString("storm.rest_metric.port"), nil)
}

func sendLatency(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		log.Println("server: error method send latency")
		return
	}

	var data RequestData
	decoder := json.NewDecoder(r.Body)
	err := decoder.Decode(&data)
	if err != nil {
		log.Println("server: error bad request send latency")
		return
	}
	latency = data.Latency
}

func GetLatency() float64 {
	return latency
}
