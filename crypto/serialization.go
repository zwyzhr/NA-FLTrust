package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/ring"
	"io"
)

// Lattigo v5.0.7's metadata BinarySize assumes a two-digit exponent.
func serializeCiphertext(ct *rlwe.Ciphertext) ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteString("NACT1")
	write := func(data []byte) error {
		if err := binary.Write(&buf, binary.LittleEndian, uint64(len(data))); err != nil {
			return err
		}
		_, err := buf.Write(data)
		return err
	}
	meta, err := json.Marshal(ct.MetaData)
	if err != nil {
		return nil, err
	}
	if err = write(meta); err != nil {
		return nil, err
	}
	if err = binary.Write(&buf, binary.LittleEndian, uint32(len(ct.Value))); err != nil {
		return nil, err
	}
	for _, poly := range ct.Value {
		data, err := poly.MarshalBinary()
		if err != nil {
			return nil, err
		}
		if err = write(data); err != nil {
			return nil, err
		}
	}
	return buf.Bytes(), nil
}

func deserializeCiphertext(data []byte) (*rlwe.Ciphertext, error) {
	if len(data) < 5 || string(data[:5]) != "NACT1" {
		return nil, fmt.Errorf("invalid ciphertext format")
	}
	reader := bytes.NewReader(data[5:])
	read := func() ([]byte, error) {
		var n uint64
		if err := binary.Read(reader, binary.LittleEndian, &n); err != nil {
			return nil, err
		}
		if n > uint64(reader.Len()) {
			return nil, io.ErrUnexpectedEOF
		}
		out := make([]byte, n)
		_, err := io.ReadFull(reader, out)
		return out, err
	}
	meta, err := read()
	if err != nil {
		return nil, err
	}
	ct := &rlwe.Ciphertext{}
	ct.MetaData = &rlwe.MetaData{}
	if err = json.Unmarshal(meta, ct.MetaData); err != nil {
		return nil, err
	}
	var n uint32
	if err = binary.Read(reader, binary.LittleEndian, &n); err != nil {
		return nil, err
	}
	if n < 1 || n > 3 {
		return nil, fmt.Errorf("invalid ciphertext degree")
	}
	ct.Value = make([]ring.Poly, n)
	for i := range ct.Value {
		b, err := read()
		if err != nil {
			return nil, err
		}
		if err = ct.Value[i].UnmarshalBinary(b); err != nil {
			return nil, err
		}
	}
	if reader.Len() != 0 {
		return nil, fmt.Errorf("trailing bytes")
	}
	return ct, nil
}
