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
	Epsilon         float64 // ε para ε-greedy
	C               float64 // c de UCB
	Alpha           float64 // paso EMA (si UseAlpha=true)
	Gamma           float64 // descuento (si UseAlpha=false)
	UseAlpha        bool
	CooldownWindows int           // #ventanas mínimas antes de cambiar modelo
	DecisionPeriod  time.Duration // duración de la ventana de decisión
	TopK            int           // para gating (opcional)
	Weights         RewardWeights
	NormBounds      RewardNormBounds
	RandomSeed      int64
	ColdStartRound  bool // probar cada modelo al menos una vez
}

/*** ===================== Estado GLOBAL ===================== ***/

type pendingDecision struct {
	DecisionID    string
	ChosenModel   string
	MadeAt        time.Time
	CooldownUntil time.Time
}

type GlobalBanditState struct {
	Config BanditSelectorConfig
	Models []string

	// Estado del bandit (Q/N/T)
	Q map[string]float64
	N map[string]int64
	T int64

	// Decisiones pendientes (crédito diferido)
	Pending map[string]pendingDecision

	// Última decisión (para cooldown)
	LastDecision pendingDecision
	HasLast      bool

	// RNG
	rng *rand.Rand
}

var Bandit GlobalBanditState

/*** ===================== Inicialización ===================== ***/

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
		return 0
	}
	return (x - b.Min) / (b.Max - b.Min)
}
func decisionID(now time.Time) string { return fmt.Sprintf("dec_%d", now.UnixNano()) }

/*** ===================== API (llamar desde MAPE) ===================== ***/

// ChooseArm selecciona el modelo para la ventana actual y devuelve (decisionID, model).
func ChooseArm(now time.Time) (string, string) {
	// Cooldown: reusar modelo si todavía no expira
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

	// Cold-start: probar modelos no vistos primero
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

	// Selección
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

	// Registrar pendiente y cooldown
	decID := decisionID(now)
	cd := now.Add(time.Duration(Bandit.Config.CooldownWindows) * Bandit.Config.DecisionPeriod)
	p := pendingDecision{DecisionID: decID, ChosenModel: chosen, MadeAt: now, CooldownUntil: cd}
	Bandit.Pending[decID] = p
	Bandit.LastDecision = p
	Bandit.HasLast = true
	return decID, chosen
}

// UpdateOutcome aplica el crédito diferido con tus 3 señales:
// - latencyMs (ms, menor es mejor)
// - degrade (0..1, menor es mejor)
// - saving  (0..1, mayor es mejor)
func UpdateOutcome(decisionID string, latencyMs, degrade, saving float64) {
	p, ok := Bandit.Pending[decisionID]
	if !ok {
		return
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

	// Normalización (a [0,1])
	latN := clamp01(norm01(latencyMs, Bandit.Config.NormBounds.Latency))
	degN := clamp01(norm01(degrade, Bandit.Config.NormBounds.Degrade))
	savN := clamp01(norm01(saving, Bandit.Config.NormBounds.Saving))

	// Reward de control
	r := Bandit.Config.Weights.WLatency*(1.0-latN) +
		Bandit.Config.Weights.WDegrade*(1.0-degN) +
		Bandit.Config.Weights.WSaving*(savN)

	// Actualización EMA (Alpha) o descuento (Gamma)
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

// RankTopK: devuelve los top-k por score actual (UCB: Q+bonus; ε-greedy: Q)
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

func BanditDefaultConfig() BanditSelectorConfig {
	return BanditSelectorConfig{
		Algorithm:       AlgoUCB, // o AlgoEpsilon
		Epsilon:         0.1,     // si usas ε-greedy
		C:               2.0,     // UCB bonus
		Alpha:           0.1,     // EMA (si UseAlpha=true)
		Gamma:           0.98,    // descuento (si UseAlpha=false)
		UseAlpha:        true,
		CooldownWindows: 2,
		DecisionPeriod:  5 * time.Second,
		TopK:            5,
		Weights: RewardWeights{
			WLatency: 0.34, WDegrade: 0.33, WSaving: 0.33,
		},
		NormBounds: RewardNormBounds{
			Latency: Bounds{Min: 50, Max: 500},   // ajusta a tu app
			Degrade: Bounds{Min: 0.0, Max: 0.25}, // SLA 10% (recomendado)
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

func UpdateBandit(topology storm.Topology) {
	maxSavedResources := viper.GetInt64("storm.adaptive.saved_resources_metric")
	var totalReplicas int64 = 0
	for _, bolt := range topology.Bolts {
		totalReplicas += bolt.Replicas
	}
	savedResourcesS := float64(totalReplicas) / float64(maxSavedResources)
	samplesBandit.SavedResources = append(samplesBandit.SavedResources, savedResourcesS)

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

	samplesBandit.Latency = append(samplesBandit.Latency, topology.Latency)
}

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
		/*latencyMs*/ latency,
		/*throughputDegradation*/ throughputDegradation,
		/*savedResources*/ savedResources)

	//log.Printf("[t=X] update bandit: savedResources={%.2f}", savedResources)
	//log.Printf("[t=X] update bandit: throughputDegradation={%.2f}", throughputDegradation)
	//log.Printf("[t=X] update bandit: latency={%.2f}", latency)

	samplesBandit.SavedResources = []float64{}
	samplesBandit.ThroughputDegradation = []float64{}
	samplesBandit.Latency = []float64{}
}
