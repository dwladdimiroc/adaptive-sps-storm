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

type Algorithm string

const (
	AlgoUCB     Algorithm = "ucb"
	AlgoEpsilon Algorithm = "epsilon"
)

type Bounds struct{ Min, Max float64 }

type RewardWeights struct {
	WLatency float64
	WDegrade float64
	WSaving  float64
}

type RewardNormBounds struct {
	Latency Bounds
	Degrade Bounds
	Saving  Bounds
}

type BanditSelectorConfig struct {
	Algorithm       Algorithm
	Epsilon         float64       // ε parameter for ε-greedy
	C               float64       // c parameter for UCB
	Alpha           float64       // EMA step size (used if UseAlpha = true)
	Gamma           float64       // Discount factor (used if UseAlpha = false)
	UseAlpha        bool          // Whether to use EMA (true) or discounted update (false)
	CooldownWindows int           // Minimum number of windows before switching models
	DecisionPeriod  time.Duration // Duration of each decision window
	TopK            int           // Used for gating (optional)
	Weights         RewardWeights // Weight coefficients for each reward signal
	NormBounds      RewardNormBounds
	RandomSeed      int64
	ColdStartRound  bool // Try each model at least once before learning begins
}

type pendingDecision struct {
	DecisionID    string
	ChosenModel   string
	MadeAt        time.Time
	CooldownUntil time.Time
}

type GlobalBanditState struct {
	Config BanditSelectorConfig
	Models []string

	// Bandit internal state (Q-values, counts, total rounds)
	Q map[string]float64
	N map[string]int64
	T int64

	// Pending decisions (for delayed credit assignment)
	Pending map[string]pendingDecision

	// Last decision (used for cooldown logic)
	LastDecision pendingDecision
	HasLast      bool

	// Random number generator
	rng *rand.Rand
}

var Bandit GlobalBanditState

// InitBandit initializes the global bandit state with the given models and configuration.
func InitBandit(models []string, cfg BanditSelectorConfig) {
	Bandit = GlobalBanditState{
		Config:  cfg,
		Models:  append([]string(nil), models...),
		Q:       make(map[string]float64, len(models)),
		N:       make(map[string]int64, len(models)),
		T:       0,
		Pending: make(map[string]pendingDecision),
		HasLast: false,
		rng:     rand.New(rand.NewSource(ifZeroSeed(cfg.RandomSeed))),
	}
	for _, m := range Bandit.Models {
		Bandit.Q[m] = 0.0
		Bandit.N[m] = 0
	}
}

// ifZeroSeed returns the given seed or generates one from the current time if seed == 0.
func ifZeroSeed(s int64) int64 {
	if s == 0 {
		return time.Now().UnixNano()
	}
	return s
}

// clamp01 clamps a value into the [0, 1] range.
func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

// norm01 normalizes a value to [0,1] based on the given bounds.
func norm01(x float64, b Bounds) float64 {
	if b.Max <= b.Min {
		return 0
	}
	return (x - b.Min) / (b.Max - b.Min)
}

// decisionID returns a unique decision identifier based on current timestamp.
func decisionID(now time.Time) string { return fmt.Sprintf("dec_%d", now.UnixNano()) }

// ChooseArm selects a model for the current decision window.
// Returns (decisionID, chosenModel).
func ChooseArm(now time.Time) (string, string) {
	// --- Cooldown phase: reuse the previous model if cooldown is still active ---
	if Bandit.HasLast && now.Before(Bandit.LastDecision.CooldownUntil) {
		decID := decisionID(now)
		p := pendingDecision{
			DecisionID:    decID,
			ChosenModel:   Bandit.LastDecision.ChosenModel,
			MadeAt:        now,
			CooldownUntil: Bandit.LastDecision.CooldownUntil,
		}
		Bandit.Pending[decID] = p
		return decID, p.ChosenModel
	}

	// --- Cold-start phase: try unseen models first ---
	if Bandit.Config.ColdStartRound {
		for _, m := range Bandit.Models {
			if Bandit.N[m] == 0 {
				Bandit.T++
				decID := decisionID(now)
				cd := now.Add(time.Duration(Bandit.Config.CooldownWindows) * Bandit.Config.DecisionPeriod)
				p := pendingDecision{DecisionID: decID, ChosenModel: m, MadeAt: now, CooldownUntil: cd}
				Bandit.Pending[decID] = p
				Bandit.LastDecision = p
				Bandit.HasLast = true
				return decID, m
			}
		}
	}

	// --- Model selection ---
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
			// Exploration
			chosen = Bandit.Models[Bandit.rng.Intn(len(Bandit.Models))]
		} else {
			// Exploitation
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

	// --- Register pending decision and apply cooldown ---
	decID := decisionID(now)
	cd := now.Add(time.Duration(Bandit.Config.CooldownWindows) * Bandit.Config.DecisionPeriod)
	p := pendingDecision{DecisionID: decID, ChosenModel: chosen, MadeAt: now, CooldownUntil: cd}
	Bandit.Pending[decID] = p
	Bandit.LastDecision = p
	Bandit.HasLast = true
	return decID, chosen
}

// UpdateOutcome applies delayed credit assignment based on three metrics:
// - latencyMs: request latency in milliseconds (lower is better)
// - degrade: throughput degradation ratio [0..1] (lower is better)
// - saving: resource savings ratio [0..1] (higher is better)
func UpdateOutcome(decisionID string, latencyMs, degrade, saving float64) {
	p, ok := Bandit.Pending[decisionID]
	if !ok {
		return
	}
	delete(Bandit.Pending, decisionID)

	// Sanitize input values
	degrade = clamp01(degrade)
	saving = clamp01(saving)

	// Normalize to [0,1]
	latN := clamp01(norm01(latencyMs, Bandit.Config.NormBounds.Latency))
	degN := clamp01(norm01(degrade, Bandit.Config.NormBounds.Degrade))
	savN := clamp01(norm01(saving, Bandit.Config.NormBounds.Saving))

	// Compute overall reward
	r := Bandit.Config.Weights.WLatency*(1.0-latN) +
		Bandit.Config.Weights.WDegrade*(1.0-degN) +
		Bandit.Config.Weights.WSaving*(savN)

	// Update Q-value using EMA (Alpha) or discounted (Gamma) update
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
}

// RankTopK returns the top-k models based on their current scores.
// For UCB, score = Q + bonus; for ε-greedy, score = Q.
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

// BanditDefaultConfig returns a default configuration for the bandit.
func BanditDefaultConfig() BanditSelectorConfig {
	return BanditSelectorConfig{
		Algorithm:       AlgoUCB, // or AlgoEpsilon
		Epsilon:         0.1,     // exploration rate (ε-greedy)
		C:               2.0,     // UCB exploration bonus coefficient
		Alpha:           0.1,     // EMA step (if UseAlpha = true)
		Gamma:           0.98,    // discount factor (if UseAlpha = false)
		UseAlpha:        true,
		CooldownWindows: 2,
		DecisionPeriod:  5 * time.Second,
		TopK:            5,
		Weights: RewardWeights{
			WLatency: 0.34, WDegrade: 0.33, WSaving: 0.33,
		},
		NormBounds: RewardNormBounds{
			Latency: Bounds{Min: 50, Max: 500},   // adjust based on your app
			Degrade: Bounds{Min: 0.0, Max: 0.25}, // recommended SLA 10%
			Saving:  Bounds{Min: 0.0, Max: 0.50},
		},
		RandomSeed:     42,
		ColdStartRound: true,
	}
}

type StatsBandit struct {
	SavedResources        []float64
	ThroughputDegradation []float64
	Latency               []float64
}

var samplesBandit StatsBandit

// UpdateBandit collects raw metric samples from the Storm topology.
func UpdateBandit(topology storm.Topology) {
	maxSavedResources := viper.GetInt64("storm.adaptive.saved_resources_metric")

	// Compute normalized resource savings
	var totalReplicas int64 = 0
	for _, bolt := range topology.Bolts {
		totalReplicas += bolt.Replicas
	}
	savedResourcesS := float64(totalReplicas) / float64(maxSavedResources)
	samplesBandit.SavedResources = append(samplesBandit.SavedResources, savedResourcesS)

	// Compute throughput degradation
	var output int64
	for _, bolt := range topology.Bolts {
		if bolt.Name == "Latency" {
			output = bolt.Output
		}
	}
	var throughputDegradationS float64
	if topology.InputRateT == 0 {
		throughputDegradationS = 1.0
	} else {
		throughputDegradationS = math.Abs(float64(topology.InputRateT)-float64(output)) / float64(topology.InputRateT)
	}
	samplesBandit.ThroughputDegradation = append(samplesBandit.ThroughputDegradation, throughputDegradationS)

	// Record latency
	samplesBandit.Latency = append(samplesBandit.Latency, topology.Latency)
}

// UpdateStatsBandit aggregates collected samples and updates the bandit outcome.
func UpdateStatsBandit(decId string) {
	var savedResources float64
	for _, sample := range samplesBandit.SavedResources {
		savedResources += sample
	}
	savedResources /= float64(len(samplesBandit.SavedResources))

	var throughputDegradation float64
	for _, sample := range samplesBandit.ThroughputDegradation {
		throughputDegradation += sample
	}
	throughputDegradation /= float64(len(samplesBandit.ThroughputDegradation))

	var latency float64
	for _, sample := range samplesBandit.Latency {
		latency += sample
	}
	latency /= float64(len(samplesBandit.Latency))

	UpdateOutcome(decId,
		/* latencyMs */ latency,
		/* throughputDegradation */ throughputDegradation,
		/* savedResources */ savedResources)

	// Reset collected samples
	samplesBandit.SavedResources = []float64{}
	samplesBandit.ThroughputDegradation = []float64{}
	samplesBandit.Latency = []float64{}
}
