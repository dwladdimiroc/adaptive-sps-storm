package predictive

import (
	"bytes"
	"encoding/json"
	"github.com/spf13/viper"
	"io"
	"log"
	"net/http"
	"strings"
)

type Response struct {
	AvgPrediction float64   `json:"avg_prediction"`
	Predictions   []float64 `json:"predictions"`
}

type PredictorData struct {
	Samples          []float64 `json:"samples"`
	PredictionNumber int       `json:"prediction_number"`
}

const PredictorURL = "http://PREDICTOR_HOST:PREDICTOR_PORT/PREDICTOR_MODEL"

func parseURL(urlRaw string, predictorModel string) string {
	var url string
	predictorHost := viper.GetString("predictor.host")
	predictorPort := viper.GetString("predictor.port")
	url = strings.Replace(urlRaw, "PREDICTOR_MODEL", predictorModel, 1)
	url = strings.Replace(url, "PREDICTOR_HOST", predictorHost, 1)
	url = strings.Replace(url, "PREDICTOR_PORT", predictorPort, 1)

	//log.Printf("Parse url %v %v\n", predictorModel, url)

	return url
}

func GetPrediction(samples []float64, predictionNumber int, predictorModel string) []float64 {
	var resp Response

	var body = PredictorData{
		Samples:          samples,
		PredictionNumber: predictionNumber,
	}

	if b, err := json.Marshal(body); err != nil {
		log.Printf("storm get prediction: %v\n", err)
	} else {
		predictor := parseURL(PredictorURL, predictorModel)
		if res, err := http.Post(predictor, "application/json", bytes.NewBuffer(b)); err != nil {
			log.Printf("storm get prediction: %v\n", err)
		} else {
			data, _ := io.ReadAll(res.Body)
			if err := res.Body.Close(); err != nil {
				log.Printf("storm get prediction: %v\n", err)
			} else {
				if err := json.Unmarshal(data, &resp); err != nil {
					log.Printf("storm get prediction: %v\n", err)
				}
			}
		}
	}

	return resp.Predictions
}
