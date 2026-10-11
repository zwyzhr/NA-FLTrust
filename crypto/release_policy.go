package main

import "crypto/rand"

type roundRejection struct {
	Code    string
	Message string
}

func (r roundRejection) Error() string { return r.Message }
func checkFillRandom(b []byte)         { _, err := rand.Read(b); check(err) }
