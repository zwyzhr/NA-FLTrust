package main

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"math"
	"math/big"
	"os"
)

type auditRequest struct {
	Slots, Limbs, Degree, Index int
	Round, Commitment           string
	SeedCommitment              string
	Seed                        []byte
	ClientKeys                  [][]byte
	Manifests                   []clientSubmission
	Root                        []float64
	Input                       clientSubmission
	Terms                       [][]byte
	Proof, Outputs              []string
	Active                      []int
}
type auditRound struct {
	Seed           []byte
	Commitment     string
	Manifests      []clientSubmission
	Root           []float64
	Degree         int
	Approved       bool
	SeedCommitment string
}
type auditAuthorization struct {
	Epoch, Round, Commitment string
	Outputs                  []string
	Active                   []int
}

func (w *holderWorker) auditHandle(r roleRequest) (out roleReply) {
	a := r.Audit
	switch r.Op {
	case "audit_init":
		var file publicKeyFile
		check(json.Unmarshal(r.Data, &file))
		w.auditFiles = []publicKeyFile{file}
		precision := uint(53)
		scale := rlwe.NewScale(math.Exp2(40))
		if w.paper {
			precision = 384
			scale = rlwe.NewScale(new(big.Int).Lsh(big.NewInt(1), 300))
		}
		w.auditEngine = &Engine{p: w.p, slots: a.Slots, limbs: a.Limbs, scale: scale, paper: w.paper, encoder: hefloat.NewEncoder(w.p, precision)}
		w.auditRounds = map[string]*auditRound{}
	case "audit_galois":
		var file publicKeyFile
		check(json.Unmarshal(r.Data, &file))
		w.auditFiles = append(w.auditFiles, file)
	case "audit_ready":
		if len(w.auditFiles) == 0 {
			panic("missing audit keys")
		}
	case "audit_clients":
		w.clientKeys = extendClientKeys(w.clientKeys, a.ClientKeys)
	case "audit_begin":
		if a.Round == "" || len(a.Manifests) == 0 || w.auditRounds[a.Round] != nil {
			panic("invalid audit round")
		}
		seen := map[int]bool{}
		for _, s := range a.Manifests {
			if s.ID < 0 || s.ID >= len(w.clientKeys) || seen[s.ID] || s.Epoch != w.epoch || s.Round != a.Round || s.Dimension != len(a.Root) || !ed25519.Verify(w.clientKeys[s.ID], clientMessage(s), s.Signature) {
				panic("committee input authentication failed")
			}
			seen[s.ID] = true
		}
		state := &auditRound{Seed: freshSeed(), Manifests: a.Manifests, Root: a.Root, Degree: a.Degree}
		state.SeedCommitment = a.SeedCommitment
		if a.SeedCommitment == "" {
			state.SeedCommitment = fmt.Sprintf("%x", sha256.Sum256(state.Seed))
		}
		if state.SeedCommitment == "" {
			panic("missing challenge commitment")
		}
		w.auditRounds[a.Round] = state
		out.Data = []byte(fmt.Sprintf("%x", sha256.Sum256(state.Seed)))
	case "audit_reveal":
		state := w.auditRounds[a.Round]
		if state == nil || state.Commitment != "" || a.Commitment == "" {
			panic("audit commitment already fixed or missing")
		}
		state.Commitment = a.Commitment
		out.Data = state.Seed
	case "audit_verify":
		state := w.auditRounds[a.Round]
		if state == nil || state.Commitment == "" || state.Approved || w.aborted[a.Round] {
			panic("invalid audit state")
		}
		if fmt.Sprintf("%x", sha256.Sum256(a.Seed)) != state.SeedCommitment {
			panic("challenge seed commitment mismatch")
		}
		if a.Index != auditIndex(a.Seed, len(state.Manifests)) {
			panic("incorrect audit challenge")
		}
		s := a.Input
		expected := state.Manifests[a.Index]
		if string(clientMessage(s)) != string(clientMessage(expected)) || !ed25519.Verify(w.clientKeys[expected.ID], clientMessage(s), s.Signature) {
			panic("opened input manifest mismatch")
		}
		inputs := []*rlwe.Ciphertext{}
		for _, data := range s.Blocks {
			ct, err := deserializeCiphertext(data)
			check(err)
			inputs = append(inputs, ct)
		}
		if digest(inputs) != expected.Digest {
			panic("authenticated input hash mismatch")
		}
		terms := []*rlwe.Ciphertext{}
		for _, data := range a.Terms {
			ct, err := deserializeCiphertext(data)
			check(err)
			terms = append(terms, ct)
		}
		leaf := digest(terms)
		if !verifyMerkle(leaf, a.Index, a.Proof, state.Commitment) {
			panic("committee Merkle verification failed")
		}
		w.loadAuditKeys()
		defer func() { w.auditEngine.eval = nil; w.auditKeys = nil }()
		t, m := w.auditEngine.trustAndNorm(inputs, state.Root, state.Degree)
		if digest([]*rlwe.Ciphertext{t, m}) != digest(terms) {
			panic("committee deterministic recomputation failed")
		}
		if len(a.Outputs) != len(inputs)+1 || len(a.Active) != w.t {
			panic("incorrect aggregate release shape")
		}
		auth := auditAuthorization{Epoch: w.epoch, Round: a.Round, Commitment: state.Commitment, Outputs: a.Outputs, Active: a.Active}
		out.Data = encode(auth)
		out.Signature = ed25519.Sign(w.signing, out.Data)
		state.Approved = true
		state.Root = nil
		state.Manifests = nil
		state.Seed = nil
	default:
		panic("unknown audit operation")
	}
	return
}

func (w *holderWorker) verifyApproval(r roleRequest) {
	if w.id >= len(w.roster) || !ed25519.Verify(w.roster[w.id].Verify, r.Authorization, r.Approval) {
		panic("release lacks committee audit approval")
	}
	var a auditAuthorization
	check(json.Unmarshal(r.Authorization, &a))
	h := r.Header
	if a.Epoch != w.epoch || a.Round != h.Round || string(encode(a.Active)) != string(encode(h.Active)) {
		panic("audit approval binding mismatch")
	}
	valid := false
	for i, hash := range a.Outputs {
		if h.Output == fmt.Sprintf("aggregate_%d", i) && h.Ciphertext == hash {
			valid = true
		}
	}
	if !valid {
		panic("output not authorized for release")
	}
}

// Public evaluation keys are pinned once and loaded only while this member
// audits, keeping q copies of large evaluation keys out of resident memory.
type publicKeyFile struct{ Path, Hash string }

func (f publicKeyFile) Read() []byte {
	data, err := os.ReadFile(f.Path)
	check(err)
	if fmt.Sprintf("%x", sha256.Sum256(data)) != f.Hash {
		panic("public evaluation key changed")
	}
	return data
}
func (w *holderWorker) loadAuditKeys() {
	rk := rlwe.NewRelinearizationKey(w.p)
	check(rk.UnmarshalBinary(w.auditFiles[0].Read()))
	keys := rlwe.NewMemEvaluationKeySet(rk)
	for _, file := range w.auditFiles[1:] {
		key := rlwe.NewGaloisKey(w.p)
		check(key.UnmarshalBinary(file.Read()))
		keys.GaloisKeys[key.GaloisElement] = key
	}
	w.auditKeys = keys
	w.auditEngine.eval = hefloat.NewEvaluator(w.p, keys)
}
