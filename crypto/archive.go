package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Epoch-scoped disk storage preserves exact retries without retaining past
// vectors or RNS shares in RAM. A process restart always creates fresh keys.
type diskCache[T any] struct{ directory string }

func newDiskCache[T any]() *diskCache[T] {
	base := os.Getenv("NAFLTRUST_CACHE_DIR")
	if base == "" {
		base = ".runtime_cache"
	}
	check(os.MkdirAll(base, 0700))
	directory, err := os.MkdirTemp(base, "nafltrust-epoch-")
	check(err)
	directory, err = filepath.Abs(directory)
	check(err)
	return &diskCache[T]{directory}
}
func (c *diskCache[T]) path(key string) string {
	return filepath.Join(c.directory, fmt.Sprintf("%x.json", sha256.Sum256([]byte(key))))
}
func (c *diskCache[T]) Save(key string, value T) {
	data, err := json.Marshal(value)
	check(err)
	target := c.path(key)
	tmp := target + ".tmp"
	check(os.WriteFile(tmp, data, 0600))
	check(os.Rename(tmp, target))
}
func (c *diskCache[T]) Load(key string) (value T, found bool) {
	data, err := os.ReadFile(c.path(key))
	if os.IsNotExist(err) {
		return value, false
	}
	check(err)
	check(json.Unmarshal(data, &value))
	return value, true
}
func (c *diskCache[T]) Close() {
	if c != nil {
		os.RemoveAll(c.directory)
	}
}

// An already pinned ID cannot be removed or rebound within a key epoch.
func extendClientKeys(old, next [][]byte) [][]byte {
	if len(next) < len(old) {
		panic("client identity registry cannot shrink")
	}
	for i, key := range next {
		if len(key) != 0 && len(key) != 32 {
			panic("invalid client verification key")
		}
		if i < len(old) && len(old[i]) > 0 && string(old[i]) != string(key) {
			panic("client identity key changed")
		}
	}
	return next
}
