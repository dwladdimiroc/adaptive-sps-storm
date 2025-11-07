package predictive

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/dwladdimiroc/sps-storm/internal/storm"
	"github.com/spf13/viper"
)

/*** ===================== Types & Configuration ===================== ***/

type Algorithm string

const (
	AlgoUCB     Algorithm = "ucb"
	AlgoEpsilon Algorithm = "epsilon"
)

type Bounds struct{ Min, Max float64 }

type RewardWeights struct {
	WLatency float64 // weight for (1 - latency_norm)
	WDegrade float64 // weight for (1 - degrade_norm)
	WSaving  float64 // weight for saving_norm
}

type RewardNormBounds struct {
	Latency Bounds // ms   (e.g., [50, 500])
	Degrade Bounds // 0..1 (e.g., [0, 0.10] if SLA allows 10% degradation)
	Saving  Bounds // 0..1 (e.g., [0, 0.50])
}

type BanditSelectorConfig struct {
	Algorithm       Algorithm
	Epsilon         float64 // ε for ε-greedy
	C               float64 // c for UCB bonus
	Alpha           float64 // EMA step (if UseAlpha=true)
	Gamma           float64 // discount factor (if UseAlpha=false)
	UseAlpha        bool
	CooldownWindows int           // not used if you switch every cycle (keep 0)
	DecisionPeriod  time.Duration // optional: for tracing/telemetry only
	TopK            int           // optional: gating
	Weights         RewardWeights
	NormBounds      RewardNormBounds
	RandomSeed      int64
	ColdStartRound  bool // try each model at least once at startup
}

/*** ===================== GLOBAL State (single-thread) ===================== ***/

type pendingDecision struct {
	DecisionID    string
	ChosenModel   string
	MadeAt        time.Time
	CooldownUntil time.Time // unused if CooldownWindows=0
}

type GlobalBanditState struct {
	// Config & model catalog
	Config BanditSelectorConfig
	Models []string

	// Bandit state (Q/N/T)
	Q map[string]float64
	N map[string]int64
	T int64

	// Pending decisions (deferred credit)
	Pending map[string]pendingDecision

	// Last decision taken (for tracing)
	LastDecision pendingDecision
	HasLast      bool

	// One open decision per window
	CurrentOpenID string
	HasOpen       bool

	// RNG
	rng *rand.Rand
}

// GLOBAL variable (assumes single-threaded use from the MAPE loop)
var Bandit GlobalBanditState

/*** ===================== Initialization ===================== ***/

func InitBandit(models []string, cfg BanditSelectorConfig) {
	Bandit = GlobalBanditState{
		Config:        cfg,
		Models:        append([]string(nil), models...),
		Q:             make(map[string]float64, len(models)),
		N:             make(map[string]int64, len(models)),
		T:             0,
		Pending:       make(map[string]pendingDecision),
		HasLast:       false,
		CurrentOpenID: "",
		HasOpen:       false,
		rng:           rand.New(rand.NewSource(ifZeroSeed(cfg.RandomSeed))),
	}
	for _, m := range Bandit.Models {
		Bandit.Q[m] = 0.0
		Bandit.N[m] = 0
	}
}

func ifZeroSeed(s int64) int64 {
	if s == 0 {
		return time.Now().UnixNano()
	}
	return s
}

/*** ===================== Helpers ===================== ***/

func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}
func norm01(x float64, b Bounds) float64 {
	if b.Max <= b.Min {
		return 0 // avoid division by zero; will saturate to 0
	}
	return (x - b.Min) / (b.Max - b.Min)
}
func decisionID(now time.Time) string { return fmt.Sprintf("dec_%d", now.UnixNano()) }

/*** ===================== API for the MAPE loop ===================== ***/

// ChooseArm: picks a model for the current "window".
// Rule: DO NOT open a new decision if one is already open (close it with UpdateOutcome first).
func ChooseArm(now time.Time) (string, string) {
	// If a decision is already open, don't open another one
	if Bandit.HasOpen {
		return Bandit.CurrentOpenID, Bandit.LastDecision.ChosenModel
	}

	// Cold-start: try unseen models first (if enabled)
	if Bandit.Config.ColdStartRound {
		for _, m := range Bandit.Models {
			if Bandit.N[m] == 0 {
				Bandit.T++
				decID := decisionID(now)
				p := pendingDecision{DecisionID: decID, ChosenModel: m, MadeAt: now}
				Bandit.Pending[decID] = p
				Bandit.LastDecision = p
				Bandit.HasLast = true
				Bandit.CurrentOpenID = decID
				Bandit.HasOpen = true
				return decID, m
			}
		}
	}

	// Selection (UCB or ε-greedy)
	var chosen string
	switch Bandit.Config.Algorithm {
	case AlgoUCB:
		Bandit.T++
		best := math.Inf(-1)
		t := math.Max(1, float64(Bandit.T))
		for _, m := range Bandit.Models {
			n := float64(Bandit.N[m])
			bonus := Bandit.Config.C * math.Sqrt(math.Log(t)/(n+1.0))
			score := Bandit.Q[m] + bonus
			if score > best {
				best = score
				chosen = m
			}
		}
	case AlgoEpsilon:
		Bandit.T++
		if Bandit.rng.Float64() < Bandit.Config.Epsilon {
			chosen = Bandit.Models[Bandit.rng.Intn(len(Bandit.Models))]
		} else {
			best := math.Inf(-1)
			for _, m := range Bandit.Models {
				if Bandit.Q[m] > best {
					best = Bandit.Q[m]
					chosen = m
				}
			}
		}
	default:
		panic("unknown algorithm")
	}

	// Register ONE open decision
	decID := decisionID(now)
	p := pendingDecision{DecisionID: decID, ChosenModel: chosen, MadeAt: now}
	Bandit.Pending[decID] = p
	Bandit.LastDecision = p
	Bandit.HasLast = true
	Bandit.CurrentOpenID = decID
	Bandit.HasOpen = true
	return decID, chosen
}

// UpdateOutcome: closes the window and applies deferred credit.
//   - latencyMs (ms, lower is better)
//   - degrade   (0..1, lower is better)
//   - saving    (0..1, higher is better)
func UpdateOutcome(decisionID string, latencyMs, degrade, saving float64) {
	p, ok := Bandit.Pending[decisionID]
	if !ok {
		return // decision not found or already applied
	}
	delete(Bandit.Pending, decisionID)

	if degrade < 0 {
		degrade = 0
	}
	if degrade > 1 {
		degrade = 1
	}
	if saving < 0 {
		saving = 0
	}
	if saving > 1 {
		saving = 1
	}

	latN := clamp01(norm01(latencyMs, Bandit.Config.NormBounds.Latency))
	degN := clamp01(norm01(degrade, Bandit.Config.NormBounds.Degrade))
	savN := clamp01(norm01(saving, Bandit.Config.NormBounds.Saving))

	// Control reward
	r := Bandit.Config.Weights.WLatency*(1.0-latN) +
		Bandit.Config.Weights.WDegrade*(1.0-degN) +
		Bandit.Config.Weights.WSaving*(savN)

	// Update Q/N
	m := p.ChosenModel
	oldQ := Bandit.Q[m]
	var newQ float64
	if Bandit.Config.UseAlpha {
		alpha := Bandit.Config.Alpha
		if alpha <= 0 || alpha > 1 {
			alpha = 0.1
		}
		newQ = (1.0-alpha)*oldQ + alpha*r
	} else {
		gamma := Bandit.Config.Gamma
		if gamma <= 0 || gamma >= 1 {
			gamma = 0.98
		}
		newQ = gamma*oldQ + (1.0-gamma)*r
	}
	Bandit.Q[m] = newQ
	Bandit.N[m] = Bandit.N[m] + 1

	// Close open decision
	if Bandit.HasOpen && Bandit.CurrentOpenID == decisionID {
		Bandit.HasOpen = false
		Bandit.CurrentOpenID = ""
	}
}

// RankTopK: returns the top-k by current score (UCB: Q+bonus; ε-greedy: Q)
func RankTopK(k int) []string {
	type pair struct {
		M string
		S float64
	}
	scores := make([]pair, 0, len(Bandit.Models))
	switch Bandit.Config.Algorithm {
	case AlgoUCB:
		t := math.Max(1, float64(Bandit.T))
		for _, m := range Bandit.Models {
			q := Bandit.Q[m]
			n := float64(Bandit.N[m])
			bonus := Bandit.Config.C * math.Sqrt(math.Log(t)/(n+1.0))
			scores = append(scores, pair{M: m, S: q + bonus})
		}
	case AlgoEpsilon:
		for _, m := range Bandit.Models {
			q := Bandit.Q[m]
			scores = append(scores, pair{M: m, S: q})
		}
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].S > scores[j].S })
	if k > len(scores) {
		k = len(scores)
	}
	out := make([]string, 0, k)
	for i := 0; i < k; i++ {
		out = append(out, scores[i].M)
	}
	return out
}

/*** ===================== Default Config ===================== ***/

func BanditDefaultConfig() BanditSelectorConfig {
	return BanditSelectorConfig{
		Algorithm:       AlgoUCB, // or AlgoEpsilon
		Epsilon:         0.1,     // if using ε-greedy
		C:               2.0,     // UCB bonus
		Alpha:           0.1,     // EMA (if UseAlpha=true)
		Gamma:           0.98,    // discount (if UseAlpha=false)
		UseAlpha:        true,
		CooldownWindows: 0,               // key: allow switching every cycle
		DecisionPeriod:  5 * time.Second, // informational
		TopK:            5,
		Weights:         RewardWeights{WLatency: 0.34, WDegrade: 0.33, WSaving: 0.33},
		NormBounds: RewardNormBounds{
			Latency: Bounds{Min: 50, Max: 500},   // adjust to your app
			Degrade: Bounds{Min: 0.0, Max: 0.10}, // set to your SLA (10% example)
			Saving:  Bounds{Min: 0.0, Max: 0.50},
		},
		RandomSeed:     42,
		ColdStartRound: false, // or true to force one shot per model at startup
	}
}

/*** ===================== Metrics Accumulation (Monitor) ===================== ***/

type StatsBandit struct {
	SavedResources        []float64
	ThroughputDegradation []float64
	Latency               []float64
}

var samplesBandit StatsBandit

// UpdateBandit: called frequently by the Monitor (append samples within the current window)
func UpdateBandit(topology storm.Topology) {
	// --- Saved Resources ---
	var totalReplicas int64
	for _, bolt := range topology.Bolts {
		totalReplicas += bolt.Replicas
	}
	baselineReplicas := viper.GetInt64("storm.adaptive.baseline_replicas") // set this in your YAML
	if baselineReplicas <= 0 {
		baselineReplicas = totalReplicas // safe fallback
	}
	saved := float64(baselineReplicas-totalReplicas) / float64(baselineReplicas)
	if saved < 0 {
		saved = 0
	}
	if saved > 1 {
		saved = 1
	}
	samplesBandit.SavedResources = append(samplesBandit.SavedResources, saved)

	// --- Throughput Degradation ---
	// You can parametrize the bolt providing the final "output"
	outputBoltName := viper.GetString("storm.adaptive.output_bolt_name") // if empty, sum outputs
	var output int64
	if outputBoltName != "" {
		for _, bolt := range topology.Bolts {
			if bolt.Name == outputBoltName {
				output = bolt.Output
				break
			}
		}
	} else {
		// fallback: sum outputs (if your topology is linear, the last would suffice)
		for _, bolt := range topology.Bolts {
			output += bolt.Output
		}
	}
	var degr float64
	if topology.InputRateT <= 0 {
		degr = 1.0
	} else {
		degr = math.Abs(float64(topology.InputRateT)-float64(output)) / float64(topology.InputRateT)
		if degr < 0 {
			degr = 0
		}
		if degr > 1 {
			degr = 1
		}
	}
	samplesBandit.ThroughputDegradation = append(samplesBandit.ThroughputDegradation, degr)

	// --- Latency ---
	samplesBandit.Latency = append(samplesBandit.Latency, topology.Latency)
}

/*** ===================== Window Aggregation & Close (Planner) ===================== ***/

// UpdateStatsBandit: computes window aggregates and applies UpdateOutcome.
// Call it ONCE per window (before the next ChooseArm).
func UpdateStatsBandit(decisionID string) {
	if decisionID == "" {
		return
	}
	if len(samplesBandit.SavedResources) == 0 ||
		len(samplesBandit.ThroughputDegradation) == 0 ||
		len(samplesBandit.Latency) == 0 {
		// empty window (e.g., startup)
		return
	}

	var saved, degr, lat float64

	for _, x := range samplesBandit.SavedResources {
		saved += x
	}
	saved /= float64(len(samplesBandit.SavedResources))

	for _, x := range samplesBandit.ThroughputDegradation {
		degr += x
	}
	degr /= float64(len(samplesBandit.ThroughputDegradation))

	for _, x := range samplesBandit.Latency {
		lat += x
	}
	lat /= float64(len(samplesBandit.Latency))

	UpdateOutcome(decisionID /*latencyMs*/, lat /*degrade*/, degr /*saving*/, saved)

	// clear buffers for next window
	samplesBandit.SavedResources = nil
	samplesBandit.ThroughputDegradation = nil
	samplesBandit.Latency = nil
}
