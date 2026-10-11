package main

// Isolated holder processes exchange authenticated encrypted DKG shares. Only
// public evaluation shares and signed partial decryptions reach the coordinator.
import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/ecdh"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/mhe"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/utils/sampling"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sync"
)

type roleIdentity struct {
	Exchange []byte
	Verify   []byte
}
type packet struct {
	From      int
	To        int
	Epoch     string
	Payload   []byte
	Signature []byte
}
type shareHeader struct {
	Epoch      string
	Round      string
	Output     string
	Ciphertext string
	Active     []int
	Holder     int
}
type roleRequest struct {
	Audit         auditRequest
	Authorization []byte
	Approval      []byte
	Op            string
	Params        hefloat.ParametersLiteral
	ID            int
	Threshold     int
	Count         int
	Epoch         string
	Paper         bool
	Budget        uint64
	Roster        []roleIdentity
	Packets       []packet
	Seed          []byte
	Data          []byte
	Galois        uint64
	Header        shareHeader
}
type roleReply struct {
	Error     string
	Identity  roleIdentity
	Data      []byte
	Signature []byte
	Packets   []packet
	Used      uint64
	Cached    bool
}
type roleProcess struct {
	cmd                      *exec.Cmd
	input                    io.WriteCloser
	output                   io.ReadCloser
	enc                      *json.Encoder
	dec                      *json.Decoder
	identity                 roleIdentity
	sentBytes, receivedBytes int64
}

func launchRole() *roleProcess {
	executable, err := os.Executable()
	check(err)
	cmd := exec.Command(executable, "--holder-worker")
	input, err := cmd.StdinPipe()
	check(err)
	output, err := cmd.StdoutPipe()
	check(err)
	cmd.Stderr = os.Stderr
	check(cmd.Start())
	role := &roleProcess{cmd: cmd, input: input, output: output}
	role.enc = json.NewEncoder(chunkedWriter{countingWriter{input, &role.sentBytes}})
	return role
}
func (r *roleProcess) call(req roleRequest) roleReply {
	if req.Op == "relin_one" || req.Op == "relin_two" || req.Op == "galois" {

	}
	if err := r.enc.Encode(req); err != nil {
		panic(roundRejection{"holder_dropout", err.Error()})
	}
	var out roleReply
	decoder := json.NewDecoder(countingReader{r.output, &r.receivedBytes})
	if err := decoder.Decode(&out); err != nil {
		panic(roundRejection{"holder_dropout", err.Error()})
	}
	if out.Error != "" {
		if len(req.Op) >= 6 && req.Op[:6] == "audit_" {
			panic(roundRejection{"audit_rejected", out.Error})
		}
		panic(out.Error)
	}
	return out
}
func (r *roleProcess) abort(round string) {
	defer func() { recover() }()
	r.call(roleRequest{Op: "abort", Header: shareHeader{Round: round}})
}
func (r *roleProcess) close() { r.input.Close(); r.output.Close(); r.cmd.Wait() }
func marshal(v interface{ MarshalBinary() ([]byte, error) }) []byte {
	b, err := v.MarshalBinary()
	check(err)
	return b
}
func encode(v interface{}) []byte { b, err := json.Marshal(v); check(err); return b }
func sharedRandom(seed []byte) *sampling.KeyedPRNG {
	r, err := sampling.NewKeyedPRNG(seed)
	check(err)
	return r
}
func freshSeed() []byte             { b := make([]byte, 32); checkFillRandom(b); return b }
func packetMessage(p packet) []byte { p.Signature = nil; return encode(p) }
func channelKey(private *ecdh.PrivateKey, public []byte) cipher.AEAD {
	pub, err := ecdh.X25519().NewPublicKey(public)
	check(err)
	secret, err := private.ECDH(pub)
	check(err)
	key := sha256.Sum256(secret)
	block, err := aes.NewCipher(key[:])
	check(err)
	gcm, err := cipher.NewGCM(block)
	check(err)
	return gcm
}

type holderWorker struct {
	auditEngine       *Engine
	auditKeys         *rlwe.MemEvaluationKeySet
	auditFiles        []publicKeyFile
	auditRounds       map[string]*auditRound
	clientKeys        [][]byte
	p                 hefloat.Parameters
	id, t, q          int
	epoch             string
	paper             bool
	budget, used      uint64
	exchange          *ecdh.PrivateKey
	signing           ed25519.PrivateKey
	identity          roleIdentity
	roster            []roleIdentity
	secret, ephemeral *rlwe.SecretKey
	share             mhe.ShamirSecretShare
	points            []mhe.ShamirPublicPoint
	cache             *diskCache[roleReply]
	delivered         *diskCache[[]byte]
	bindings          map[string]string
	aborted           map[string]bool
	dkgPoly           *mhe.ShamirPolynomial
	dkgSeen, dkgSent  map[int]bool
}

func (w *holderWorker) handle(r roleRequest) (out roleReply) {
	defer func() {
		if v := recover(); v != nil {
			out.Error = fmt.Sprint(v)
		}
	}()
	if len(r.Op) >= 6 && r.Op[:6] == "audit_" {
		return w.auditHandle(r)
	}
	switch r.Op {
	case "ping":
	case "init":
		var err error
		w.p, err = hefloat.NewParametersFromLiteral(r.Params)
		check(err)
		w.id = r.ID
		w.t = r.Threshold
		w.q = r.Count
		w.epoch = r.Epoch
		w.paper = r.Paper
		w.budget = r.Budget
		w.exchange, err = ecdh.X25519().GenerateKey(rand.Reader)
		check(err)
		pub, priv, err := ed25519.GenerateKey(rand.Reader)
		check(err)
		w.signing = priv
		w.identity = roleIdentity{w.exchange.PublicKey().Bytes(), pub}
		out.Identity = w.identity
		w.secret = rlwe.NewKeyGenerator(w.p).GenSecretKeyNew()
		tz := mhe.NewThresholdizer(w.p)
		w.share = tz.AllocateThresholdSecretShare()
		w.points = make([]mhe.ShamirPublicPoint, w.q)
		for i := range w.points {
			w.points[i] = mhe.ShamirPublicPoint(i + 1)
		}
		w.cache = newDiskCache[roleReply]()
		w.delivered = newDiskCache[[]byte]()
		w.bindings = map[string]string{}
		w.aborted = map[string]bool{}
	case "dkg_prepare":
		if len(w.roster) > 0 || len(r.Roster) != w.q {
			panic("invalid DKG roster")
		}
		w.roster = r.Roster
		w.dkgSeen = map[int]bool{}
		w.dkgSent = map[int]bool{}
	case "dkg_send":
		j := r.ID
		if j < 0 || j >= w.q || len(w.roster) != w.q || w.dkgSent[j] {
			panic("invalid DKG recipient")
		}
		tz := mhe.NewThresholdizer(w.p)
		if w.dkgPoly == nil {
			poly, err := tz.GenShamirPolynomial(w.t, w.secret)
			check(err)
			w.dkgPoly = &poly
		}
		share := tz.AllocateThresholdSecretShare()
		tz.GenShamirSecretShare(w.points[j], *w.dkgPoly, &share)
		aead := channelKey(w.exchange, w.roster[j].Exchange)
		p := packet{From: w.id, To: j, Epoch: w.epoch}
		nonce := make([]byte, aead.NonceSize())
		checkFillRandom(nonce)
		p.Payload = append(nonce, aead.Seal(nil, nonce, marshal(share), encode([]interface{}{p.From, p.To, p.Epoch}))...)
		p.Signature = ed25519.Sign(w.signing, packetMessage(p))
		out.Packets = []packet{p}
		w.dkgSent[j] = true
		if len(w.dkgSent) == w.q {
			w.dkgPoly = nil
		}
	case "dkg_receive":
		if len(r.Packets) != 1 {
			panic("missing DKG sender")
		}
		seen := w.dkgSeen
		tz := mhe.NewThresholdizer(w.p)
		for _, p := range r.Packets {
			if p.From < 0 || p.From >= w.q || p.To != w.id || p.Epoch != w.epoch || seen[p.From] {
				panic("invalid DKG binding")
			}
			seen[p.From] = true
			if !ed25519.Verify(w.roster[p.From].Verify, packetMessage(p), p.Signature) {
				panic("DKG authentication failed")
			}
			aead := channelKey(w.exchange, w.roster[p.From].Exchange)
			if len(p.Payload) < aead.NonceSize() {
				panic("truncated DKG packet")
			}
			plain, err := aead.Open(nil, p.Payload[:aead.NonceSize()], p.Payload[aead.NonceSize():], encode([]interface{}{p.From, p.To, p.Epoch}))
			check(err)
			part := tz.AllocateThresholdSecretShare()
			check(part.UnmarshalBinary(plain))
			check(tz.AggregateShares(w.share, part, &w.share))
		}
	case "public_key":
		if len(w.dkgSeen) != w.q {
			panic("incomplete DKG")
		}
		p := mhe.NewPublicKeyGenProtocol(w.p)
		share := p.AllocateShare()
		p.GenShare(w.secret, p.SampleCRP(sharedRandom(r.Seed)), &share)
		out.Data = marshal(share)
	case "relin_one":
		p := mhe.NewRelinearizationKeyGenProtocol(w.p)
		ep, share, _ := p.AllocateShare()
		w.ephemeral = ep
		p.GenShareRoundOne(w.secret, p.SampleCRP(sharedRandom(r.Seed)), ep, &share)
		out.Data = marshal(share)
	case "relin_two":
		p := mhe.NewRelinearizationKeyGenProtocol(w.p)
		_, aggregate, share := p.AllocateShare()
		check(aggregate.UnmarshalBinary(r.Data))
		p.GenShareRoundTwo(w.ephemeral, w.secret, aggregate, &share)
		out.Data = marshal(share)
		w.ephemeral = nil
	case "galois":
		p := mhe.NewGaloisKeyGenProtocol(w.p)
		share := p.AllocateShare()
		check(p.GenShare(w.secret, r.Galois, p.SampleCRP(sharedRandom(r.Seed)), &share))
		out.Data = marshal(share)
	case "abort":
		w.aborted[r.Header.Round] = true
		if state := w.auditRounds[r.Header.Round]; state != nil {
			state.Root = nil
			state.Manifests = nil
			state.Seed = nil
		}
	case "aggregate_receive":
		w.verifyApproval(r)
		if fmt.Sprintf("%x", sha256.Sum256(r.Data)) != r.Header.Ciphertext {
			panic("aggregate delivery hash mismatch")
		}
		w.delivered.Save(r.Header.Round+":"+r.Header.Output, r.Data)
	case "decrypt":
		h := r.Header
		if len(r.Data) == 0 {
			var exists bool
			r.Data, exists = w.delivered.Load(h.Round + ":" + h.Output)
			if !exists {
				panic("aggregate not delivered")
			}
		}
		w.verifyApproval(r)
		if h.Epoch != w.epoch || h.Holder != w.id || h.Round == "" || h.Output == "" || len(h.Active) != w.t {
			panic("invalid release header")
		}
		sum := sha256.Sum256(r.Data)
		if fmt.Sprintf("%x", sum) != h.Ciphertext {
			panic("ciphertext binding mismatch")
		}
		seen := map[int]bool{}
		points := make([]mhe.ShamirPublicPoint, w.t)
		for j, i := range h.Active {
			if i < 0 || i >= w.q || seen[i] {
				panic("invalid active roster")
			}
			seen[i] = true
			points[j] = w.points[i]
		}
		if !seen[w.id] {
			panic("inactive holder")
		}
		if w.aborted[h.Round] {
			panic("aborted round")
		}
		activeBinding := string(encode(h.Active))
		if old, ok := w.bindings[h.Round]; ok && old != activeBinding {
			panic("active set changed")
		}
		w.bindings[h.Round] = activeBinding
		key := h.Round + ":" + h.Output
		binding := string(encode(h))
		if old, ok := w.bindings[key]; ok && old != binding {
			panic("release binding changed")
		}
		if cached, ok := w.cache.Load(key); ok {
			cached.Cached = true
			return cached
		}
		if uint64(w.p.N()) > w.budget-w.used {
			panic("holder release budget exhausted")
		}
		w.bindings[key] = binding
		ct, err := deserializeCiphertext(r.Data)
		check(err)
		rq := w.p.RingQ().AtLevel(ct.Level())
		sk := rlwe.NewSecretKey(w.p)
		comb := mhe.NewCombiner(*w.p.GetRLWEParameters(), w.points[w.id], w.points, w.t)
		check(comb.GenAdditiveShare(points, w.points[w.id], w.share, sk))
		share := rq.NewPoly()
		rq.MulCoeffsMontgomery(ct.Value[1], sk.Value.Q, share)
		noise := rq.NewPoly()
		if w.paper {
			rq.SetCoefficientsBigint(gaussianBig(236, w.p.N()), noise)
		} else {
			prng, err := sampling.NewPRNG()
			check(err)
			sampler, err := ring.NewSampler(prng, rq, ring.DiscreteGaussian{Sigma: 3.2, Bound: 19.2}, false)
			check(err)
			sampler.Read(noise)
		}
		rq.NTT(noise, noise)
		rq.Add(share, noise, share)
		w.used += uint64(w.p.N())
		out.Data = marshal(share)
		out.Signature = ed25519.Sign(w.signing, append(encode(h), out.Data...))
		out.Used = w.used
		w.cache.Save(key, out)
	default:
		panic("unknown holder operation")
	}
	return
}
func serveHolder() {
	worker := &holderWorker{}
	defer func() { worker.cache.Close(); worker.delivered.Close() }()
	encoder := json.NewEncoder(chunkedWriter{os.Stdout})
	for {
		var r roleRequest
		decoder := json.NewDecoder(os.Stdin)
		if err := decoder.Decode(&r); err == io.EOF {
			return
		} else {
			check(err)
		}
		reply := worker.handle(r)
		check(encoder.Encode(reply))
		// Synchronous request/reply transport: only trailing JSON whitespace can remain.
		// Discard large decoder buffers and transient encoder pools before going idle.
		big := len(r.Data) > 1048576 || len(reply.Data) > 1048576 || len(r.Packets) > 0 || len(reply.Packets) > 0 || r.Op == "audit_verify"
		r = roleRequest{}
		reply = roleReply{}
		decoder = nil
		if big {
			runtime.GC()
			debug.FreeOSMemory()
		}

	}
}

var useIsolated bool
var initialBudget uint64 = 1 << 40

func setupIsolated(p hefloat.Parameters, lit hefloat.ParametersLiteral, profile string, q, t, slots, limbs int, scale rlwe.Scale, precision uint) *Engine {
	e := &Engine{p: p, threshold: t, slots: slots, limbs: limbs, scale: scale, paper: profile != "smoke", maxDegree: 4, cache: newDiskCache[Response](), bindings: map[string]string{}, budget: initialBudget}
	ready := false
	defer func() {
		if !ready {
			for _, worker := range e.workers {
				worker.close()
			}
		}
	}()
	if profile == "paper-wide" {
		e.maxDegree = 16
	}
	e.epoch = fmt.Sprintf("%x", freshSeed())
	e.encoder = hefloat.NewEncoder(p, precision)
	roster := make([]roleIdentity, q)
	for i := 0; i < q; i++ {
		worker := launchRole()
		e.workers = append(e.workers, worker)
		reply := worker.call(roleRequest{Op: "init", Params: lit, ID: i, Threshold: t, Count: q, Epoch: e.epoch, Paper: e.paper, Budget: e.budget})
		roster[i] = reply.Identity
		worker.identity = reply.Identity
	}
	for _, worker := range e.workers {
		worker.call(roleRequest{Op: "dkg_prepare", Roster: roster})
	}
	for _, worker := range e.workers {
		for j, recipient := range e.workers {
			reply := worker.call(roleRequest{Op: "dkg_send", ID: j})
			recipient.call(roleRequest{Op: "dkg_receive", Packets: reply.Packets})
		}
	}

	pkp := mhe.NewPublicKeyGenProtocol(p)
	seed := freshSeed()
	pkcrp := pkp.SampleCRP(sharedRandom(seed))
	pkagg := pkp.AllocateShare()
	for _, worker := range e.workers {
		r := worker.call(roleRequest{Op: "public_key", Seed: seed})
		s := pkp.AllocateShare()
		check(s.UnmarshalBinary(r.Data))
		pkp.AggregateShares(pkagg, s, &pkagg)
	}
	pk := rlwe.NewPublicKey(p)
	pkp.GenPublicKey(pkagg, pkcrp, pk)
	e.publicKey = pk
	e.enc = rlwe.NewEncryptor(p, pk)

	rkp := mhe.NewRelinearizationKeyGenProtocol(p)
	seed = freshSeed()
	_, r1, r2 := rkp.AllocateShare()
	for _, worker := range e.workers {
		r := worker.call(roleRequest{Op: "relin_one", Seed: seed})
		_, s, _ := rkp.AllocateShare()
		check(s.UnmarshalBinary(r.Data))
		rkp.AggregateShares(r1, s, &r1)
	}

	first := marshal(r1)
	for _, worker := range e.workers {
		r := worker.call(roleRequest{Op: "relin_two", Data: first})
		_, _, s := rkp.AllocateShare()
		check(s.UnmarshalBinary(r.Data))
		rkp.AggregateShares(r2, s, &r2)
	}

	rk := rlwe.NewRelinearizationKey(p)
	rkp.GenRelinearizationKey(r1, r2, rk)
	rkFile := e.publicKeyReference("relin", marshal(rk))
	for _, worker := range e.workers {
		worker.call(roleRequest{Op: "audit_init", Data: rkFile, Audit: auditRequest{Slots: slots, Limbs: limbs}})
	}
	gp := mhe.NewGaloisKeyGenProtocol(p)
	gks := []*rlwe.GaloisKey{}
	for rotation := 1; rotation < slots; rotation *= 2 {

		element := p.GaloisElement(rotation)
		seed = freshSeed()
		crp := gp.SampleCRP(sharedRandom(seed))
		aggregate := gp.AllocateShare()
		aggregate.GaloisElement = element
		for _, worker := range e.workers {
			r := worker.call(roleRequest{Op: "galois", Seed: seed, Galois: element})
			s := gp.AllocateShare()
			check(s.UnmarshalBinary(r.Data))
			check(gp.AggregateShares(aggregate, s, &aggregate))
		}
		key := rlwe.NewGaloisKey(p)
		check(gp.GenGaloisKey(aggregate, crp, key))
		gks = append(gks, key)
		keyFile := e.publicKeyReference(fmt.Sprintf("rotation_%d", rotation), marshal(key))
		for _, worker := range e.workers {
			worker.call(roleRequest{Op: "audit_galois", Data: keyFile})
		}
	}
	for _, worker := range e.workers {
		worker.call(roleRequest{Op: "audit_ready"})
	}
	keys := rlwe.NewMemEvaluationKeySet(rk, gks...)
	e.eval = hefloat.NewEvaluator(p, keys)
	ready = true
	return e
}

func (e *Engine) decryptIsolated(ct *rlwe.Ciphertext, active []int) ([]float64, int) {
	data := marshalCipher(ct)
	hash := sha256.Sum256(data)
	rq := e.p.RingQ().AtLevel(ct.Level())
	total := rq.NewPoly()
	bytes := 0
	count := len(active)
	if e.dropAfter > 0 && count > e.dropAfter-e.sent {
		count = e.dropAfter - e.sent
		if count < 0 {
			count = 0
		}
	}
	type result struct {
		reply   roleReply
		header  shareHeader
		failure interface{}
	}
	results := make([]result, count)
	var group sync.WaitGroup
	// Charge every dispatched share, including uncertain delivery, before send.
	e.used += uint64(count) * uint64(e.p.N())
	e.sent += count
	for j, i := range active[:count] {
		group.Add(1)
		go func(j, i int) {
			defer group.Done()
			defer func() { results[j].failure = recover() }()
			h := shareHeader{Epoch: e.epoch, Round: e.currentRound, Output: fmt.Sprintf("aggregate_%d", e.outputIndex), Ciphertext: fmt.Sprintf("%x", hash), Active: active, Holder: i}
			results[j].header = h
			results[j].reply = e.workers[i].call(roleRequest{Op: "decrypt", Header: h, Authorization: e.authorization, Approval: e.approvals[i]})
		}(j, i)
	}
	group.Wait()
	for j, result := range results {
		if result.failure != nil {
			panic(result.failure)
		}
		reply, h := result.reply, result.header
		if !ed25519.Verify(e.workers[active[j]].identity.Verify, append(encode(h), reply.Data...), reply.Signature) {
			panic(roundRejection{"audit_rejected", "partial decryption signature mismatch"})
		}
		share := rq.NewPoly()
		check(share.UnmarshalBinary(reply.Data))
		rq.Add(total, share, total)
		bytes += len(reply.Data) + len(reply.Signature) + len(encode(h))
		if reply.Cached {
			e.used -= uint64(e.p.N())
			e.sent--
		}
	}
	if count < len(active) {
		panic(roundRejection{"holder_dropout", "holder dropped after partial release; round aborted"})
	}

	e.outputIndex++
	pt := hefloat.NewPlaintext(e.p, ct.Level())
	*pt.MetaData = *ct.MetaData
	rq.Add(ct.Value[0], total, pt.Value)
	v := make([]float64, e.slots)
	check(e.encoder.Decode(pt, v))
	return v, bytes
}
func marshalCipher(ct *rlwe.Ciphertext) []byte {
	b, err := serializeCiphertext(ct)
	check(err)
	return b
}

func (e *Engine) publicKeyReference(name string, data []byte) []byte {
	path := filepath.Join(e.cache.directory, name+".public")
	check(os.WriteFile(path, data, 0600))
	return encode(publicKeyFile{Path: path, Hash: fmt.Sprintf("%x", sha256.Sum256(data))})
}

// Bound audit memory while retaining independently executed committee checks.
func parallelRoleCalls(count, limit int, call func(int) roleReply) []roleReply {
	if limit < 1 {
		limit = 1
	}
	if limit > count {
		limit = count
	}
	replies := make([]roleReply, count)
	failures := make([]interface{}, count)
	jobs := make(chan int)
	var group sync.WaitGroup
	for worker := 0; worker < limit; worker++ {
		group.Add(1)
		go func() {
			defer group.Done()
			for i := range jobs {
				func() { defer func() { failures[i] = recover() }(); replies[i] = call(i) }()
			}
		}()
	}
	for i := 0; i < count; i++ {
		jobs <- i
	}
	close(jobs)
	group.Wait()
	for _, failure := range failures {
		if failure != nil {
			panic(failure)
		}
	}
	return replies
}
