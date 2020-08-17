// +build noasm !amd64

package convolver

func add(a, b, r *kernelWeight) {
	r.R = a.R + b.R
	r.G = a.G + b.G
	r.B = a.B + b.B
	r.A = a.A + b.A
	return
}

func mul(a, b, r *kernelWeight) {
	r.R = a.R * b.R
	r.G = a.G * b.G
	r.B = a.B * b.B
	r.A = a.A * b.A
	return
}

func maxOne(a, b float32) float32 {
	if a >= b {
		return a
	}

	return b
}

func max(a, b, r *kernelWeight) {
	r.R = maxOne(a.R, b.R)
	r.G = maxOne(a.G, b.G)
	r.B = maxOne(a.B, b.B)
	r.A = maxOne(a.A, b.A)
	return
}

func minOne(a, b float32) float32 {
	if a <= b {
		return a
	}

	return b
}

func min(a, b, r *kernelWeight) {
	r.R = minOne(a.R, b.R)
	r.G = minOne(a.G, b.G)
	r.B = minOne(a.B, b.B)
	r.A = minOne(a.A, b.A)
	return
}
