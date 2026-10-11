package main

import (
	"crypto/sha256"
	"encoding/binary"
	"math/big"
)

// Rejection sampling avoids the modulo bias of mapping a 256-bit hash directly to n.
func auditIndex(seed []byte, n int) int {
	if n < 1 {
		panic("empty audit population")
	}
	modulus := big.NewInt(int64(n))
	space := new(big.Int).Lsh(big.NewInt(1), 256)
	limit := new(big.Int).Sub(space, new(big.Int).Mod(new(big.Int).Set(space), modulus))
	for counter := uint64(0); ; counter++ {
		payload := append([]byte{}, seed...)
		suffix := make([]byte, 8)
		binary.LittleEndian.PutUint64(suffix, counter)
		payload = append(payload, suffix...)
		hash := sha256.Sum256(payload)
		value := new(big.Int).SetBytes(hash[:])
		if value.Cmp(limit) < 0 {
			return int(value.Mod(value, modulus).Int64())
		}
	}
}
