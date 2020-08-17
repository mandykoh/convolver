// +build !noasm,amd64

package convolver

//go:noescape
//go:nosplit
func add(a, b, result *kernelWeight)

//go:noescape
//go:nosplit
func mul(a, b, result *kernelWeight)

//go:noescape
//go:nosplit
func max(a, b, result *kernelWeight)

//go:noescape
//go:nosplit
func min(a, b, result *kernelWeight)
