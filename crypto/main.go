// NA-FLTrust CKKS execution harness with isolated client and holder processes.
package main

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/mhe"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/utils/bignum"
	"github.com/tuneinsight/lattigo/v5/utils/sampling"
	"io"
	"math"
	"math/big"
	"os"
	"time"
)

func check(e error) {
	if e != nil {
		panic(e)
	}
}

type Engine struct {
	publicKey               *rlwe.PublicKey
	authorization, approval []byte
	approvals               [][]byte
	clientKeys              [][]byte
	workers                 []*roleProcess
	live                    []int
	auditWorkers            int
	currentRound            string
	outputIndex             int
	epoch                   string
	budget                  uint64
	used                    uint64
	dropAfter               int
	sent                    int
	p                       hefloat.Parameters
	enc                     *rlwe.Encryptor
	encoder                 *hefloat.Encoder
	eval                    *hefloat.Evaluator
	keys                    []*rlwe.SecretKey
	threshold               int
	limbs                   int
	scale                   rlwe.Scale
	paper                   bool
	maxDegree               int
	slots                   int
	cache                   *diskCache[Response]
	bindings                map[string]string
}
type Request struct {
	Encrypted  []clientSubmission `json:"encrypted,omitempty"`
	ClientKeys [][]byte           `json:"client_keys,omitempty"`
	DropAfter  int                `json:"drop_after,omitempty"`
	Session    string             `json:"session"`
	Updates    [][]float64        `json:"updates"`
	Root       []float64          `json:"root"`
	Degree     int                `json:"degree"`
	Active     []int              `json:"active"`
	Tamper     bool               `json:"tamper"`
}
type Response struct {
	HolderPIDs              []int     `json:"holder_pids,omitempty"`
	ErrorCode               string    `json:"error_code,omitempty"`
	Epoch                   string    `json:"epoch"`
	CoefficientsUsed        uint64    `json:"coefficients_used"`
	Numerator               []float64 `json:"numerator"`
	Denominator             float64   `json:"denominator"`
	Seconds                 float64   `json:"seconds"`
	Level                   int       `json:"output_level"`
	InputBytes              int       `json:"input_bytes"`
	OutputBytes             int       `json:"output_bytes"`
	Audit                   bool      `json:"audit_passed"`
	AuditIndex              int       `json:"audit_index"`
	Commitment              string    `json:"commitment"`
	SeedCommitment          string    `json:"seed_commitment"`
	AuditSeed               string    `json:"audit_seed"`
	Cached                  bool      `json:"cached"`
	Error                   string    `json:"error,omitempty"`
	Profile                 string    `json:"profile"`
	AuditMembers            []int     `json:"audit_members"`
	PrincipalInputBytes     int64     `json:"principal_input_bytes"`
	PrincipalAuditBytes     int64     `json:"principal_audit_bytes"`
	PrincipalOutputBytes    int64     `json:"principal_aggregate_delivery_bytes"`
	PrincipalShareBytes     int64     `json:"principal_share_bytes"`
	PrincipalPublicBytes    int64     `json:"principal_public_vector_bytes"`
	PrincipalBytesPerClient float64   `json:"principal_bytes_per_client"`
	CommitteeWireBytes      int64     `json:"committee_json_wire_bytes"`
	ShareBytes              int       `json:"share_bytes"`
}

func setup(profile string, q, t, slots int) *Engine {
	if q < 2 || t < 2 || t > q {
		panic("require 2 <= threshold <= holders")
	}
	lit := hefloat.ParametersLiteral{LogN: 14, LogQ: []int{60, 40, 40, 40, 40, 40}, LogP: []int{60}, LogDefaultScale: 40}
	limbs := 1
	scale := rlwe.NewScale(math.Exp2(40))
	prec := uint(53)
	if profile == "paper" || profile == "paper-wide" {
		lit.LogN = 16
		count := 26
		if profile == "paper-wide" {
			lit.LogN = 17
			count = 36
		}
		lit.LogQ = make([]int, count)
		for i := range lit.LogQ {
			lit.LogQ[i] = 60
		}
		limbs = 5
		scale = rlwe.NewScale(new(big.Int).Lsh(big.NewInt(1), 300))
		prec = 384
	} else if profile != "smoke" {
		panic("profile must be smoke, paper, or paper-wide")
	}
	p, err := hefloat.NewParametersFromLiteral(lit)
	check(err)
	if slots == 0 {
		slots = p.MaxSlots()
	}
	if slots < 1 || slots > p.MaxSlots() || slots&(slots-1) != 0 {
		panic("slots must be a power of two no larger than MaxSlots")
	}
	if useIsolated {
		return setupIsolated(p, lit, profile, q, t, slots, limbs, scale, prec)
	}
	kg := rlwe.NewKeyGenerator(p)
	crs, err := sampling.NewPRNG()
	check(err)
	secrets := make([]*rlwe.SecretKey, q)
	shares := make([]mhe.ShamirSecretShare, q)
	points := make([]mhe.ShamirPublicPoint, q)
	tz := mhe.NewThresholdizer(p)
	for i := range secrets {
		secrets[i] = kg.GenSecretKeyNew()
		shares[i] = tz.AllocateThresholdSecretShare()
		points[i] = mhe.ShamirPublicPoint(i + 1)
	}
	for _, sk := range secrets {
		poly, err := tz.GenShamirPolynomial(t, sk)
		check(err)
		for j := range shares {
			s := tz.AllocateThresholdSecretShare()
			tz.GenShamirSecretShare(points[j], poly, &s)
			check(tz.AggregateShares(shares[j], s, &shares[j]))
		}
	}
	// Collective public and evaluation keys: share protocols, never sum secret keys.
	pkp := mhe.NewPublicKeyGenProtocol(p)
	pkagg := pkp.AllocateShare()
	pkcrp := pkp.SampleCRP(crs)
	for _, sk := range secrets {
		s := pkp.AllocateShare()
		pkp.GenShare(sk, pkcrp, &s)
		pkp.AggregateShares(pkagg, s, &pkagg)
	}
	pk := rlwe.NewPublicKey(p)
	pkp.GenPublicKey(pkagg, pkcrp, pk)
	rkp := mhe.NewRelinearizationKeyGenProtocol(p)
	_, r1, r2 := rkp.AllocateShare()
	rcrp := rkp.SampleCRP(crs)
	eph := make([]*rlwe.SecretKey, q)
	for i, sk := range secrets {
		ep, s, _ := rkp.AllocateShare()
		eph[i] = ep
		rkp.GenShareRoundOne(sk, rcrp, ep, &s)
		rkp.AggregateShares(r1, s, &r1)
	}
	for i, sk := range secrets {
		_, _, s := rkp.AllocateShare()
		rkp.GenShareRoundTwo(eph[i], sk, r1, &s)
		rkp.AggregateShares(r2, s, &r2)
	}
	rk := rlwe.NewRelinearizationKey(p)
	rkp.GenRelinearizationKey(r1, r2, rk)
	gp := mhe.NewGaloisKeyGenProtocol(p)
	gks := []*rlwe.GaloisKey{}
	for rot := 1; rot < slots; rot *= 2 {
		el := p.GaloisElement(rot)
		crp := gp.SampleCRP(crs)
		agg := gp.AllocateShare()
		agg.GaloisElement = el
		for _, sk := range secrets {
			s := gp.AllocateShare()
			check(gp.GenShare(sk, el, crp, &s))
			check(gp.AggregateShares(agg, s, &agg))
		}
		key := rlwe.NewGaloisKey(p)
		check(gp.GenGaloisKey(agg, crp, key))
		gks = append(gks, key)
	}
	e := &Engine{p: p, enc: rlwe.NewEncryptor(p, pk), encoder: hefloat.NewEncoder(p, prec), eval: hefloat.NewEvaluator(p, rlwe.NewMemEvaluationKeySet(rk, gks...)), threshold: t, slots: slots, limbs: limbs, scale: scale, paper: profile == "paper" || profile == "paper-wide", maxDegree: 4, cache: newDiskCache[Response](), bindings: map[string]string{}}
	e.publicKey = pk
	if profile == "paper-wide" {
		e.maxDegree = 16
	}
	thresholdState[e] = state{shares, points}
	epochBytes := make([]byte, 32)
	checkFillRandom(epochBytes)
	e.epoch = hex.EncodeToString(epochBytes)
	e.budget = 1 << 40
	return e
}

type state struct {
	shares []mhe.ShamirSecretShare
	points []mhe.ShamirPublicPoint
}

var thresholdState = map[*Engine]state{}

func (e *Engine) rescale(x *rlwe.Ciphertext) {
	level := x.Level()
	if level < e.limbs {
		panic("insufficient modulus")
	}
	rq := e.p.RingQ().AtLevel(level)
	buf := rq.NewPoly()
	for i := 0; i < e.limbs; i++ {
		x.Scale = x.Scale.Div(rlwe.NewScale(e.p.Q()[level-i]))
	}
	for _, v := range x.Value {
		rq.DivRoundByLastModulusManyNTT(e.limbs, v, buf, v)
	}
	x.Resize(x.Degree(), level-e.limbs)
}
func (e *Engine) mul(a, b *rlwe.Ciphertext) *rlwe.Ciphertext {
	z, err := e.eval.MulRelinNew(a, b)
	check(err)
	e.rescale(z)
	return z
}
func (e *Engine) add(a, b *rlwe.Ciphertext) *rlwe.Ciphertext {
	if a == nil {
		return b.CopyNew()
	}
	z, err := e.eval.AddNew(a, b)
	check(err)
	return z
}
func (e *Engine) sum(x *rlwe.Ciphertext) *rlwe.Ciphertext {
	z := x.CopyNew()
	for r := 1; r < e.slots; r *= 2 {
		v, err := e.eval.RotateNew(z, r)
		check(err)
		check(e.eval.Add(z, v, z))
	}
	return z
}
func (e *Engine) encrypt(v []float64) *rlwe.Ciphertext {
	pt := hefloat.NewPlaintext(e.p, e.p.MaxLevel())
	pt.Scale = e.scale
	pt.LogDimensions.Cols = int(math.Log2(float64(e.slots)))
	check(e.encoder.Encode(v, pt))
	ct, err := e.enc.EncryptNew(pt)
	check(err)
	return ct
}
func (e *Engine) trustAndNorm(blocks []*rlwe.Ciphertext, g []float64, k int) (*rlwe.Ciphertext, *rlwe.Ciphertext) {
	var sim, norm *rlwe.Ciphertext
	slots := e.slots
	for j, ct := range blocks {
		vals := make([]float64, slots)
		for z := 0; z < slots && j*slots+z < len(g); z++ {
			vals[z] = g[j*slots+z] / 2
		}
		pt := hefloat.NewPlaintext(e.p, ct.Level())
		pt.Scale = e.scale
		pt.LogDimensions.Cols = int(math.Log2(float64(e.slots)))
		check(e.encoder.Encode(vals, pt))
		v, err := e.eval.MulNew(ct, pt)
		check(err)
		e.rescale(v)
		sim = e.add(sim, e.sum(v))
		norm = e.add(norm, e.sum(e.mul(ct, ct)))
	}
	check(e.eval.Add(sim, .5, sim))
	w := sim
	cache := map[int]*rlwe.Ciphertext{} // At most two distinct exponents per recursion level.
	var power func(*rlwe.Ciphertext, int) *rlwe.Ciphertext
	power = func(x *rlwe.Ciphertext, n int) *rlwe.Ciphertext {
		if saved, ok := cache[n]; ok {
			return saved.CopyNew()
		}
		var result *rlwe.Ciphertext
		if n == 1 {
			result = x.CopyNew()
		} else {
			left := power(x, n/2)
			if n%2 == 0 {
				result = e.mul(left, left)
			} else {
				result = e.mul(left, power(x, n-n/2))
			}
		}
		cache[n] = result.CopyNew()
		return result
	}
	w = power(w, k)
	return w, norm
}
func (e *Engine) terms(blocks []*rlwe.Ciphertext, g []float64, k int) ([]*rlwe.Ciphertext, *rlwe.Ciphertext, *rlwe.Ciphertext, *rlwe.Ciphertext) {
	w, norm := e.trustAndNorm(blocks, g, k)
	out := make([]*rlwe.Ciphertext, len(blocks))
	for j, ct := range blocks {
		out[j] = e.mul(w, ct)
	}
	return out, e.mul(w, norm), w, norm
}
func digest(cts []*rlwe.Ciphertext) string {
	h := sha256.New()
	for _, ct := range cts {
		b, err := serializeCiphertext(ct)
		check(err)
		h.Write(b)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// Sampler-error certification is separate from numerical validation.
var gaussianGrid = map[int]*big.Int{}

func gaussianGridCut(index int) *big.Int {
	if cached, ok := gaussianGrid[index]; ok {
		return cached
	}
	numerator := new(big.Int).Mul(big.NewInt(int64(index)), big.NewInt(int64(index)))
	exponent := new(big.Float).SetPrec(448).Quo(new(big.Float).SetPrec(448).SetInt(numerator), new(big.Float).SetPrec(448).SetInt64(2*1024*1024))
	exponent.Neg(exponent)
	probability := bignum.Exp(exponent)
	probability.Mul(probability, new(big.Float).SetPrec(448).SetInt(new(big.Int).Lsh(big.NewInt(1), 384)))
	cut, _ := probability.Int(nil)
	gaussianGrid[index] = cut
	return cut
}
func gaussianBig(logSigma int, n int) []*big.Int {
	sigma := new(big.Int).Lsh(big.NewInt(1), uint(logSigma))
	bound := new(big.Int).Mul(sigma, big.NewInt(20))
	width := new(big.Int).Add(new(big.Int).Lsh(new(big.Int).Set(bound), 1), big.NewInt(1))
	twoSigma2 := new(big.Int).Lsh(new(big.Int).Mul(sigma, sigma), 1)
	coinRange := new(big.Int).Lsh(big.NewInt(1), 384)
	out := make([]*big.Int, n)
	for i := 0; i < n; {
		u, err := rand.Int(rand.Reader, width)
		check(err)
		x := new(big.Int).Sub(u, bound)
		coin, err := rand.Int(rand.Reader, coinRange)
		check(err)
		index := int(new(big.Int).Quo(new(big.Int).Lsh(new(big.Int).Abs(x), 10), sigma).Int64())
		// Monotonic probability bounds reject/accept most proposals without another Exp.
		// One-integer padding encloses the grid cutoff rounding. Ambiguous proposals
		// use the same 448-bit calculation as the unsqueezed rejection sampler.
		lower := new(big.Int).Sub(gaussianGridCut(index+1), big.NewInt(1))
		upper := new(big.Int).Add(gaussianGridCut(index), big.NewInt(1))
		if coin.Cmp(lower) < 0 {
			out[i] = x
			i++
			continue
		}
		if coin.Cmp(upper) >= 0 {
			continue
		}
		square := new(big.Int).Mul(x, x)
		exponent := new(big.Float).SetPrec(448).Quo(new(big.Float).SetPrec(448).SetInt(square), new(big.Float).SetPrec(448).SetInt(twoSigma2))
		exponent.Neg(exponent)
		probability := bignum.Exp(exponent)
		probability.Mul(probability, new(big.Float).SetPrec(448).SetInt(coinRange))
		cut, _ := probability.Int(nil)
		if coin.Cmp(cut) < 0 {
			out[i] = x
			i++
		}
	}
	return out
}

func (e *Engine) decrypt(ct *rlwe.Ciphertext, active []int) ([]float64, int) {
	if len(e.workers) > 0 {
		return e.decryptIsolated(ct, active)
	}
	st := thresholdState[e]
	pts := make([]mhe.ShamirPublicPoint, len(active))
	for j, i := range active {
		pts[j] = st.points[i]
	}
	rq := e.p.RingQ().AtLevel(ct.Level())
	total := rq.NewPoly()
	bytes := 0
	for _, i := range active {
		if e.dropAfter > 0 && e.sent >= e.dropAfter {
			panic(roundRejection{"holder_dropout", "holder dropped after partial release; round aborted"})
		}
		sk := rlwe.NewSecretKey(e.p)
		comb := mhe.NewCombiner(*e.p.GetRLWEParameters(), st.points[i], st.points, e.threshold)
		check(comb.GenAdditiveShare(pts, st.points[i], st.shares[i], sk))
		share := rq.NewPoly()
		rq.MulCoeffsMontgomery(ct.Value[1], sk.Value.Q, share)
		noise := rq.NewPoly()
		if e.paper {
			rq.SetCoefficientsBigint(gaussianBig(236, e.p.N()), noise)
		} else {
			prng, err := sampling.NewPRNG()
			check(err)
			sampler, err := ring.NewSampler(prng, rq, ring.DiscreteGaussian{Sigma: 3.2, Bound: 19.2}, false)
			check(err)
			sampler.Read(noise)
		}
		rq.NTT(noise, noise)
		rq.Add(share, noise, share)
		e.used += uint64(e.p.N())
		e.sent++
		rq.Add(total, share, total)
		bytes += 8 * e.p.N() * (ct.Level() + 1)
	}
	pt := hefloat.NewPlaintext(e.p, ct.Level())
	*pt.MetaData = *ct.MetaData
	rq.Add(ct.Value[0], total, pt.Value)
	v := make([]float64, e.slots)
	check(e.encoder.Decode(pt, v))
	return v, bytes
}
func (e *Engine) run(req Request) (out Response) {
	defer func() {
		for _, worker := range e.workers {
			out.HolderPIDs = append(out.HolderPIDs, worker.cmd.Process.Pid)
		}
		out.Epoch = e.epoch
		out.CoefficientsUsed = e.used
		if v := recover(); v != nil {
			for _, worker := range e.workers {
				worker.abort(req.Session)
			}
			if rejection, ok := v.(roundRejection); ok {
				out.ErrorCode = rejection.Code
				out.Error = rejection.Message
				return
			}
			out.Error = fmt.Sprint(v)
		}
	}()
	start := time.Now()
	wireBefore := e.committeeWireBytes()
	defer func() { out.CommitteeWireBytes = e.committeeWireBytes() - wireBefore }()
	profile := "smoke"
	if e.paper {
		profile = "paper"
		if e.maxDegree == 16 {
			profile = "paper-wide"
		}
	}
	out.Profile = profile
	if len(req.Encrypted) > 0 {
		if len(req.Updates) > 0 {
			panic("do not mix plaintext and encrypted submissions")
		}
		req.Updates = make([][]float64, len(req.Encrypted))
	} else if len(e.workers) > 0 {
		panic("isolated aggregation requires encrypted client submissions")
	}
	if req.Session == "" || len(req.Updates) == 0 || len(req.Root) == 0 {
		panic("session, updates and root required")
	}
	if req.Degree == 0 {
		req.Degree = 4
	}
	if req.Degree < 2 || req.Degree%2 != 0 {
		panic("positive even degree required")
	}
	if req.Degree > e.maxDegree {
		panic("degree exceeds the configured modulus depth budget")
	}
	if len(e.workers) > 0 {
		e.live = nil
		for i, worker := range e.workers {
			alive := func() (ok bool) { defer func() { recover() }(); worker.call(roleRequest{Op: "ping"}); return true }()
			if alive {
				e.live = append(e.live, i)
			}
		}
		if len(e.live) < e.threshold {
			panic(roundRejection{"insufficient_holders", "fewer than threshold live holders before release"})
		}
		if len(req.Active) == 0 {
			req.Active = append([]int{}, e.live[:e.threshold]...)
		}
		for _, id := range req.Active {
			found := false
			for _, live := range e.live {
				if id == live {
					found = true
				}
			}
			if !found {
				panic(roundRejection{"insufficient_holders", "requested active holder unavailable before shares"})
			}
		}
	} else if len(req.Active) == 0 {
		for i := 0; i < e.threshold; i++ {
			req.Active = append(req.Active, i)
		}
	}
	if len(req.Active) != e.threshold {
		panic(roundRejection{"insufficient_holders", "exactly threshold holders must be fixed before release"})
	}
	seen := map[int]bool{}
	for _, i := range req.Active {
		count := len(thresholdState[e].points)
		if len(e.workers) > 0 {
			count = len(e.workers)
		}
		if i < 0 || i >= count || seen[i] {
			panic("invalid active set")
		}
		seen[i] = true
	}
	payload, err := json.Marshal(req)
	check(err)
	binding := fmt.Sprintf("%x", sha256.Sum256(payload))
	if old, ok := e.bindings[req.Session]; ok {
		if old != binding {
			panic("session already bound to different input or active set")
		}
		cached, exists := e.cache.Load(req.Session)
		if !exists {
			panic("session previously aborted; use fresh round and ciphertexts")
		}
		out = cached
		out.Cached = true
		return
	}
	e.bindings[req.Session] = binding // reserve before any shares, including failed rounds
	d := len(req.Root)
	radius := 0.
	for _, v := range req.Root {
		radius += v * v
	}
	radius = math.Sqrt(radius)
	if radius == 0 || math.IsNaN(radius) || math.IsInf(radius, 0) {
		panic("zero or nonfinite root")
	}
	g := make([]float64, d)
	for j := range g {
		g[j] = req.Root[j] / radius
	}
	slots := e.slots
	nb := (d + slots - 1) / slots
	maxNorm := 0.0
	if len(req.Encrypted) > 0 {
		maxNorm = 16
		if e.maxDegree == 16 {
			maxNorm = 1
		}
	}
	for _, u := range req.Updates {
		squared := 0.0
		for _, v := range u {
			squared += v * v
		}
		maxNorm = math.Max(maxNorm, math.Sqrt(squared))
	}
	depth := 2 + int(math.Ceil(math.Log2(float64(req.Degree))))
	remaining := e.p.MaxLevel() + 1 - depth*e.limbs
	room := -math.Log2(e.scale.Float64())
	for _, q := range e.p.Q()[:remaining] {
		room += math.Log2(float64(q))
	}
	bound := math.Log2(float64(len(req.Updates))) + float64(req.Degree)*math.Log2((maxNorm+1)/2) + math.Max(math.Log2(maxNorm), 2*math.Log2(maxNorm))
	if e.maxDegree == 16 && maxNorm > 1.000001 {
		panic("paper-wide profile is reserved for unit-input degree sensitivity")
	}
	if math.IsNaN(maxNorm) || math.IsInf(maxNorm, 0) || bound+8 > room {
		panic("input magnitude exceeds this profile's conservative output integer budget")
	}
	inputs := make([][]*rlwe.Ciphertext, len(req.Updates))
	numTerms := make([][]*rlwe.Ciphertext, len(inputs))
	trustTerms := make([]*rlwe.Ciphertext, len(inputs))
	normTerms := make([]*rlwe.Ciphertext, len(inputs))
	seed := make([]byte, 32)
	_, err = rand.Read(seed)
	check(err)
	out.SeedCommitment = fmt.Sprintf("%x", sha256.Sum256(seed))
	if len(e.workers) > 0 {
		manifests := append([]clientSubmission{}, req.Encrypted...)
		for i := range manifests {
			manifests[i].Blocks = nil
		}
		r := e.workers[e.live[0]].call(roleRequest{Op: "audit_begin", Audit: auditRequest{Round: req.Session, Manifests: manifests, Root: g, Degree: req.Degree}})
		out.SeedCommitment = string(r.Data)
		for _, id := range e.live[1:] {
			e.workers[id].call(roleRequest{Op: "audit_begin", Audit: auditRequest{Round: req.Session, Manifests: manifests, Root: g, Degree: req.Degree, SeedCommitment: out.SeedCommitment}})
		}
	}
	denTerms := make([]*rlwe.Ciphertext, len(inputs))
	leaves := []string{}
	clientDigests := make([]string, len(inputs))
	for i, u := range req.Updates {
		if len(req.Encrypted) == 0 && len(u) != d {
			panic("dimension mismatch")
		}
		for _, v := range u {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				panic("nonfinite input")
			}
		}
		inputs[i] = make([]*rlwe.Ciphertext, nb)
		for b := 0; b < nb; b++ {
			end := (b + 1) * slots
			if end > d {
				end = d
			}
			var ct *rlwe.Ciphertext
			if len(req.Encrypted) > 0 {
				s := req.Encrypted[i]
				if s.ID < 0 || s.ID >= len(e.clientKeys) || s.Epoch != e.epoch || s.Round != req.Session || s.Dimension != d || len(s.Blocks) != nb {
					panic(roundRejection{"audit_rejected", "invalid signed client manifest"})
				}
				if !ed25519.Verify(e.clientKeys[s.ID], clientMessage(s), s.Signature) {
					panic(roundRejection{"audit_rejected", "client authentication failed"})
				}
				ct, err = deserializeCiphertext(s.Blocks[b])
				check(err)
			} else {
				ct = e.encrypt(u[b*slots : end])
			}
			inputs[i][b] = ct
			buf, err := serializeCiphertext(ct)
			check(err)
			out.InputBytes += len(buf)
			out.PrincipalInputBytes += principalCipherBytes(ct)
		}
		clientDigests[i] = digest(inputs[i]) // checked against the client-signed manifest
		if len(req.Encrypted) > 0 && clientDigests[i] != req.Encrypted[i].Digest {
			panic(roundRejection{"audit_rejected", "signed input digest mismatch"})
		}
		numTerms[i], denTerms[i], trustTerms[i], normTerms[i] = e.terms(inputs[i], g, req.Degree)
		leaves = append(leaves, digest([]*rlwe.Ciphertext{trustTerms[i], normTerms[i]}))
	}

	out.Commitment = merkleRoot(leaves) // fixed transcript commitment before cryptographic random audit challenge
	if len(e.workers) > 0 {
		r := e.workers[e.live[0]].call(roleRequest{Op: "audit_reveal", Audit: auditRequest{Round: req.Session, Commitment: out.Commitment}})
		seed = r.Data
		for _, id := range e.live[1:] {
			e.workers[id].call(roleRequest{Op: "audit_reveal", Audit: auditRequest{Round: req.Session, Commitment: out.Commitment}})
		}
	}
	out.AuditSeed = hex.EncodeToString(seed)
	idx := auditIndex(seed, len(inputs))
	out.AuditIndex = idx
	if req.Tamper {
		check(e.eval.Add(trustTerms[idx], 1., trustTerms[idx]))
	}
	if digest(inputs[idx]) != clientDigests[idx] {
		panic("authenticated client digest mismatch")
	}
	proof := merkleProof(leaves, idx)
	current := digest([]*rlwe.Ciphertext{trustTerms[idx], normTerms[idx]})
	if len(e.workers) == 0 && !verifyMerkle(current, idx, proof, out.Commitment) {
		panic(roundRejection{"audit_rejected", "Merkle proof mismatch; no shares released"})
	}
	if len(e.workers) == 0 {
		_, _, at, am := e.terms(inputs[idx], g, req.Degree)
		if digest([]*rlwe.Ciphertext{at, am}) != digest([]*rlwe.Ciphertext{trustTerms[idx], normTerms[idx]}) {
			panic(roundRejection{"audit_rejected", "sampled evaluation mismatch; no shares released"})
		}
	}
	out.Audit = len(e.workers) == 0
	needed := uint64(nb+1) * uint64(e.threshold) * uint64(e.p.N())
	if needed > e.budget {
		panic(roundRejection{"budget_too_small", "configured epoch budget cannot fit one round"})
	}
	if e.used > e.budget-needed {
		panic(roundRejection{"budget_exhausted", "refresh key epoch before release"})
	}
	e.dropAfter = req.DropAfter
	e.sent = 0
	e.currentRound = req.Session
	e.outputIndex = 0
	nums := make([]*rlwe.Ciphertext, nb)
	var den *rlwe.Ciphertext
	for i := range inputs {
		for b := range nums {
			nums[b] = e.add(nums[b], numTerms[i][b])
		}
		den = e.add(den, denTerms[i])
	}
	if len(e.workers) > 0 {
		outputs := []string{}
		for _, ct := range append(nums, den) {
			h := sha256.Sum256(marshalCipher(ct))
			outputs = append(outputs, fmt.Sprintf("%x", h))
		}
		terms := [][]byte{}
		for _, ct := range []*rlwe.Ciphertext{trustTerms[idx], normTerms[idx]} {
			terms = append(terms, marshalCipher(ct))
		}
		e.approvals = make([][]byte, len(e.workers))
		auditReplies := parallelRoleCalls(len(e.live), e.auditWorkers, func(position int) roleReply {
			i := e.live[position]
			return e.workers[i].call(roleRequest{Op: "audit_verify", Audit: auditRequest{Round: req.Session, Index: idx, Proof: proof, Input: req.Encrypted[idx], Terms: terms, Outputs: outputs, Active: req.Active, Seed: seed}})
		})
		for position, r := range auditReplies {
			i := e.live[position]
			if position > 0 && string(e.authorization) != string(r.Data) {
				panic("committee authorization disagreement")
			}
			e.authorization = r.Data
			e.approvals[i] = r.Signature
			out.AuditMembers = append(out.AuditMembers, i)
		}
		auditPrincipal := principalCipherBytes(trustTerms[idx]) + principalCipherBytes(normTerms[idx])
		for _, ct := range inputs[idx] {
			auditPrincipal += principalCipherBytes(ct)
		}
		out.PrincipalAuditBytes = int64(len(e.live)) * auditPrincipal
		// Deliver each aggregate to every member exactly once, including
		// non-active holders. Active members load their delivered copy.
		for output, ct := range append(nums, den) {
			data := marshalCipher(ct)
			for _, i := range e.live {
				worker := e.workers[i]
				worker.call(roleRequest{Op: "aggregate_receive", Data: data, Authorization: e.authorization, Approval: e.approvals[i], Header: shareHeader{Epoch: e.epoch, Round: req.Session, Output: fmt.Sprintf("aggregate_%d", output), Ciphertext: outputs[output], Active: req.Active, Holder: i}})
				out.PrincipalOutputBytes += principalCipherBytes(ct)
			}
			out.PrincipalShareBytes += int64(e.threshold) * int64(e.p.N()) * 8 * int64(ct.Level()+1)
		}
		out.Audit = true
	}

	out.Numerator = make([]float64, 0, d)
	for b, ct := range nums {
		v, n := e.decrypt(ct, req.Active)
		out.ShareBytes += n
		end := slots
		if d-b*slots < end {
			end = d - b*slots
		}
		out.Numerator = append(out.Numerator, v[:end]...)
		buf, err := serializeCiphertext(ct)
		check(err)
		out.OutputBytes += len(buf)
	}
	v, n := e.decrypt(den, req.Active)
	out.ShareBytes += n
	out.Denominator = v[0]
	out.Level = den.Level()
	buf, err := serializeCiphertext(den)
	check(err)
	out.OutputBytes += len(buf)
	out.PrincipalPublicBytes = int64(4 * d * (2*len(inputs) + len(e.live)))
	out.PrincipalBytesPerClient = float64(out.PrincipalInputBytes+out.PrincipalAuditBytes+out.PrincipalOutputBytes+out.PrincipalShareBytes+out.PrincipalPublicBytes) / float64(len(inputs))
	out.Seconds = time.Since(start).Seconds()
	e.cache.Save(req.Session, out)
	return
}
func main() {
	client := flag.Bool("client-worker", false, "isolated client encryption process")
	worker := flag.Bool("holder-worker", false, "isolated private holder process")
	isolated := flag.Bool("isolated", true, "isolate holder keys and shares in separate processes")
	profile := flag.String("profile", "paper", "smoke: functional low precision; paper: N=65536,26x60,scale=2^300 and 2^236 smudging")
	q := flag.Int("holders", 20, "committee size")
	t := flag.Int("threshold", 14, "release threshold")
	slots := flag.Int("slots", 0, "logical packing slots; 0 uses full capacity")
	auditWorkers := flag.Int("audit-workers", 2, "maximum concurrent independent committee audits")
	budget := flag.Uint64("release-budget", 1<<40, "maximum fresh share coefficient instances per key epoch")
	flag.Parse()
	if *worker {
		serveHolder()
		return
	}
	if *client {
		serveClient()
		return
	}
	useIsolated = *isolated
	initialBudget = *budget
	e := setup(*profile, *q, *t, *slots)
	defer func() {
		e.cache.Close()
		for _, worker := range e.workers {
			worker.close()
		}
	}()
	if *auditWorkers < 1 {
		panic("audit-workers must be positive")
	}
	e.auditWorkers = *auditWorkers
	e.budget = *budget

	decoder := json.NewDecoder(os.Stdin)
	enc := json.NewEncoder(chunkedWriter{os.Stdout})
	maxNorm := 16.
	if e.maxDegree == 16 {
		maxNorm = 1
	}
	check(enc.Encode(publicSetup{MaxNorm: maxNorm, Ready: true, Params: marshal(e.p), PublicKey: marshal(e.publicKey), Slots: e.slots, Epoch: e.epoch, Paper: e.paper}))
	for {
		var req Request
		if err := decoder.Decode(&req); err == io.EOF {
			return
		} else if err != nil {
			check(enc.Encode(Response{Error: err.Error()}))
			return
		}
		if len(req.ClientKeys) > 0 {
			reply := func() (result map[string]interface{}) {
				defer func() {
					if v := recover(); v != nil {
						result = map[string]interface{}{"error": fmt.Sprint(v)}
					}
				}()
				next := extendClientKeys(e.clientKeys, req.ClientKeys)
				registered := 0
				for _, worker := range e.workers {
					ok := func() (success bool) {
						defer func() { recover() }()
						worker.call(roleRequest{Op: "audit_clients", Audit: auditRequest{ClientKeys: next}})
						return true
					}()
					if ok {
						registered++
					}
				}
				if registered < e.threshold {
					panic("fewer than threshold reachable committee members for registration")
				}
				e.clientKeys = next
				return map[string]interface{}{"registered": true}
			}()
			check(enc.Encode(reply))
			continue
		}
		check(enc.Encode(e.run(req)))
	}

}

func principalCipherBytes(ct *rlwe.Ciphertext) int64 {
	return int64(len(ct.Value)) * int64(len(ct.Value[0].Coeffs[0])) * int64(ct.Level()+1) * 8
}
func (e *Engine) committeeWireBytes() int64 {
	var n int64
	for _, w := range e.workers {
		n += w.sentBytes + w.receivedBytes
	}
	return n
}
