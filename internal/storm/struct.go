package storm

type SummaryTopologies struct {
	Topologies []struct {
		Name string `json:"name"`
		Id   string `json:"id"`
	} `json:"topologies"`

	Error string `json:"error"`
}

type SummaryTopology struct {
	Name string `json:"name"`
	Id   string `json:"id"`

	Spouts []struct {
		SpoutId string `json:"spoutId"`
	} `json:"spouts"`

	Bolts []struct {
		BoltID string `json:"boltId"`
	} `json:"bolts"`

	Error string `json:"error"`
}

type TopologyMetrics struct {
	Spouts []SpoutMetrics `json:"spouts"`
	Bolts  []BoltMetrics  `json:"bolts"`
}

type BoltMetrics struct {
	Id string `json:"id"`

	InputStats []struct {
		Component string `json:"component"` //InputId
	} `json:"inputStats"`

	BoltStats []struct {
		ExecuteLatency string `json:"executeLatency"`
		Window         string `json:"window"`
		Executed       int64  `json:"executed"`
	} `json:"boltStats"`

	OutputStats []struct {
		Emitted int64  `json:"emitted"`
		Stream  string `json:"stream"`
	} `json:"outputStats"`
}

type SpoutMetrics struct {
	Id string `json:"id"`

	SpoutSummary []struct {
		Emitted         int     `json:"emitted"`
		CompleteLatency float64 `json:"completeLatency"`
		Window          string  `json:"window"` //:all-time
	} `json:"spoutSummary"`

	OutputStats []struct {
		Emitted         int    `json:"emitted"`
		CompleteLatency string `json:"completeLatency"`
		Stream          string `json:"stream"` // Bolt Id
	} `json:"outputStats"`
}
