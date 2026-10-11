package main

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"io"
	"math"
	"math/big"
	"os"
	"runtime"
	"runtime/debug"
)

type clientSubmission struct {
	ID        int
	Epoch     string
	Round     string
	Dimension int
	Blocks    [][]byte
	Digest    string
	Signature []byte
}

func clientMessage(s clientSubmission) []byte { s.Blocks = nil; s.Signature = nil; return encode(s) }

type publicSetup struct {
	MaxNorm   float64 `json:"max_norm"`
	Ready     bool    `json:"ready"`
	Params    []byte  `json:"params"`
	PublicKey []byte  `json:"public_key"`
	Slots     int     `json:"slots"`
	Epoch     string  `json:"epoch"`
	Paper     bool    `json:"paper"`
}
type clientCommand struct {
	Setup  *publicSetup `json:"setup,omitempty"`
	ID     int          `json:"id"`
	Round  string       `json:"round"`
	Values []float64    `json:"values"`
}

func serveClient() {
	encoder := json.NewEncoder(chunkedWriter{os.Stdout})
	var engine *Engine
	var signing ed25519.PrivateKey
	var id int
	var epoch string
	var maxNorm float64
	for {
		decoder := json.NewDecoder(os.Stdin)
		var request clientCommand
		if err := decoder.Decode(&request); err == io.EOF {
			return
		} else {
			check(err)
		}
		if request.Setup != nil {
			cfg := request.Setup
			maxNorm = cfg.MaxNorm
			var p hefloat.Parameters
			check(p.UnmarshalBinary(cfg.Params))
			pk := rlwe.NewPublicKey(p)
			check(pk.UnmarshalBinary(cfg.PublicKey))
			precision := uint(53)
			scale := rlwe.NewScale(math.Exp2(40))
			if cfg.Paper {
				precision = 384
				scale = rlwe.NewScale(new(big.Int).Lsh(big.NewInt(1), 300))
			}
			engine = &Engine{p: p, slots: cfg.Slots, scale: scale, encoder: hefloat.NewEncoder(p, precision), enc: rlwe.NewEncryptor(p, pk)}
			pub, priv, err := ed25519.GenerateKey(rand.Reader)
			check(err)
			signing = priv
			id = request.ID
			epoch = cfg.Epoch
			check(encoder.Encode(map[string]interface{}{"identity": pub}))
			continue
		}
		if engine == nil || request.Round == "" {
			panic("client not initialized")
		}
		norm := 0.
		for _, v := range request.Values {
			norm = math.Hypot(norm, v)
		}
		if math.IsNaN(norm) || math.IsInf(norm, 0) || norm > maxNorm+1e-6 {
			check(encoder.Encode(map[string]string{"error": "input exceeds configured finite magnitude domain"}))
			continue
		}
		s := clientSubmission{ID: id, Epoch: epoch, Round: request.Round, Dimension: len(request.Values)}
		cts := []*rlwe.Ciphertext{}
		for start := 0; start < len(request.Values); start += engine.slots {
			end := start + engine.slots
			if end > len(request.Values) {
				end = len(request.Values)
			}
			ct := engine.encrypt(request.Values[start:end])
			cts = append(cts, ct)
			s.Blocks = append(s.Blocks, marshalCipher(ct))
		}
		s.Digest = digest(cts)
		s.Signature = ed25519.Sign(signing, clientMessage(s))
		check(encoder.Encode(s))
		if len(request.Values) > 65536 {
			request = clientCommand{}
			s = clientSubmission{}
			cts = nil
			decoder = nil
			runtime.GC()
			debug.FreeOSMemory()
		}
	}
}
