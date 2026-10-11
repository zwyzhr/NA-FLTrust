package main

import (
	"crypto/sha256"
	"encoding/hex"
)

func leafHash(value string) string {
	h := sha256.Sum256([]byte("leaf:" + value))
	return hex.EncodeToString(h[:])
}
func parentHash(left, right string) string {
	h := sha256.Sum256([]byte("node:" + left + right))
	return hex.EncodeToString(h[:])
}
func merkleLevel(nodes []string) []string {
	next := []string{}
	for i := 0; i < len(nodes); i += 2 {
		j := i + 1
		if j == len(nodes) {
			j = i
		}
		next = append(next, parentHash(nodes[i], nodes[j]))
	}
	return next
}
func merkleLeaves(values []string) []string {
	nodes := make([]string, len(values))
	for i, v := range values {
		nodes[i] = leafHash(v)
	}
	return nodes
}
func merkleRoot(values []string) string {
	nodes := merkleLeaves(values)
	if len(nodes) == 0 {
		panic("empty Merkle tree")
	}
	for len(nodes) > 1 {
		nodes = merkleLevel(nodes)
	}
	return nodes[0]
}
func merkleProof(values []string, index int) []string {
	nodes := merkleLeaves(values)
	proof := []string{}
	for len(nodes) > 1 {
		j := index ^ 1
		if j >= len(nodes) {
			j = index
		}
		proof = append(proof, nodes[j])
		nodes = merkleLevel(nodes)
		index /= 2
	}
	return proof
}
func verifyMerkle(value string, index int, proof []string, root string) bool {
	h := leafHash(value)
	for _, sibling := range proof {
		if index%2 == 0 {
			h = parentHash(h, sibling)
		} else {
			h = parentHash(sibling, h)
		}
		index /= 2
	}
	return h == root
}
