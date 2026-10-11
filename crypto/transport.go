package main

import "io"

// Windows anonymous pipes can reject a single very large WriteFile request.
// Split JSON writes without changing framing or the signed byte representation.
type chunkedWriter struct{ io.Writer }

func (w chunkedWriter) Write(data []byte) (int, error) {
	total := 0
	for len(data) > 0 {
		size := len(data)
		if size > 65536 {
			size = 65536
		}
		n, err := w.Writer.Write(data[:size])
		total += n
		data = data[n:]
		if err != nil {
			return total, err
		}
		if n == 0 {
			return total, io.ErrShortWrite
		}
	}
	return total, nil
}

// Local transport bytes include JSON/base64 and authentication overhead.
type countingWriter struct {
	w     io.Writer
	total *int64
}

func (w countingWriter) Write(p []byte) (int, error) {
	n, e := w.w.Write(p)
	*w.total += int64(n)
	return n, e
}

type countingReader struct {
	r     io.Reader
	total *int64
}

func (r countingReader) Read(p []byte) (int, error) {
	n, e := r.r.Read(p)
	*r.total += int64(n)
	return n, e
}
