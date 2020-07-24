package convolver

import (
	"image"
	"image/color"
	"math"
	"sync"
)

type opFunc func(img *image.NRGBA, x, y int) color.NRGBA

type Kernel struct {
	radius     int
	sideLength int
	weights    []kernelWeight
}

func (k *Kernel) ApplyMax(img *image.NRGBA, parallelism int) *image.NRGBA {
	return k.apply(img, k.Max, parallelism)
}

func (k *Kernel) ApplyMin(img *image.NRGBA, parallelism int) *image.NRGBA {
	return k.apply(img, k.Min, parallelism)
}

func (k *Kernel) ApplySum(img *image.NRGBA, parallelism int) *image.NRGBA {
	return k.apply(img, k.Sum, parallelism)
}

func (k *Kernel) apply(img *image.NRGBA, op opFunc, parallelism int) *image.NRGBA {
	bounds := img.Rect
	result := image.NewNRGBA(bounds)

	var allDone sync.WaitGroup
	allDone.Add(parallelism)

	for worker := 0; worker < parallelism; worker++ {
		workerNum := worker

		go func() {
			defer allDone.Done()

			for i := bounds.Min.Y + workerNum; i < bounds.Max.Y; i += parallelism {
				for j := bounds.Min.X; j < bounds.Max.X; j++ {
					result.SetNRGBA(j, i, op(img, j, i))
				}
			}
		}()
	}

	allDone.Wait()

	return result
}

func (k *Kernel) clipToBounds(bounds image.Rectangle, x, y int) kernelClip {
	clip := kernelClip{}

	if edgeDist := x - bounds.Min.X; edgeDist < k.radius {
		clip.Left = edgeDist
	}
	if edgeDist := bounds.Max.X - x - 1; edgeDist < k.radius {
		clip.Right = edgeDist
	}
	if edgeDist := y - bounds.Min.Y; edgeDist < k.radius {
		clip.Top = edgeDist
	}
	if edgeDist := bounds.Max.Y - y - 1; edgeDist < k.radius {
		clip.Bottom = edgeDist
	}

	return clip
}

func (k *Kernel) Max(img *image.NRGBA, x, y int) color.NRGBA {
	max := kernelWeight{
		math.MinInt32,
		math.MinInt32,
		math.MinInt32,
		math.MinInt32,
	}

	for s := 0; s < k.sideLength; s++ {
		for t := 0; t < k.sideLength; t++ {
			weight := k.weights[s*k.sideLength+t]

			c := img.NRGBAAt(x+t-k.radius, y+s-k.radius)
			if int32(c.R)*weight.R > max.R*weight.R {
				max.R = int32(c.R)
			}
			if int32(c.G)*weight.G > max.G*weight.G {
				max.G = int32(c.G)
			}
			if int32(c.B)*weight.B > max.B*weight.B {
				max.B = int32(c.B)
			}
			if int32(c.A)*weight.A > max.A*weight.A {
				max.A = int32(c.A)
			}
		}
	}

	return max.toNRGBA()
}

func (k *Kernel) Min(img *image.NRGBA, x, y int) color.NRGBA {
	min := kernelWeight{
		math.MaxInt32,
		math.MaxInt32,
		math.MaxInt32,
		math.MaxInt32,
	}

	for s := 0; s < k.sideLength; s++ {
		for t := 0; t < k.sideLength; t++ {
			weight := k.weights[s*k.sideLength+t]

			c := img.NRGBAAt(x+t-k.radius, y+s-k.radius)
			if int32(c.R)*weight.R < min.R*weight.R {
				min.R = int32(c.R)
			}
			if int32(c.G)*weight.G < min.G*weight.G {
				min.G = int32(c.G)
			}
			if int32(c.B)*weight.B < min.B*weight.B {
				min.B = int32(c.B)
			}
			if int32(c.A)*weight.A < min.A*weight.A {
				min.A = int32(c.A)
			}
		}
	}

	return min.toNRGBA()
}

func (k *Kernel) SetWeight(x, y int, r, g, b, a int32) {
	k.weights[y*k.sideLength+x] = kernelWeight{
		R: r,
		G: g,
		B: b,
		A: a,
	}
}

func (k *Kernel) SetWeightRGBA(x, y int, weight int32) {
	k.SetWeight(x, y, weight, weight, weight, weight)
}

func (k *Kernel) SideLength() int {
	return k.sideLength
}

func (k *Kernel) Sum(img *image.NRGBA, x, y int) color.NRGBA {
	clip := k.clipToBounds(img.Rect, x, y)

	totalWeight := kernelWeight{}
	sum := kernelWeight{}

	for s := clip.Top; s < k.sideLength-clip.Bottom; s++ {
		for t := clip.Left; t < k.sideLength-clip.Right; t++ {
			weight := k.weights[s*k.sideLength+t]
			totalWeight.R += weight.R
			totalWeight.G += weight.G
			totalWeight.B += weight.B
			totalWeight.A += weight.A

			c := img.NRGBAAt(x+t-k.radius, y+s-k.radius)
			sum.R += int32(c.R) * weight.R
			sum.G += int32(c.G) * weight.G
			sum.B += int32(c.B) * weight.B
			sum.A += int32(c.A) * weight.A
		}
	}

	if totalWeight.R > 0 {
		sum.R /= totalWeight.R
	}
	if totalWeight.G > 0 {
		sum.G /= totalWeight.G
	}
	if totalWeight.B > 0 {
		sum.B /= totalWeight.B
	}
	if totalWeight.A > 0 {
		sum.A /= totalWeight.A
	}

	return sum.toNRGBA()
}

func KernelWithRadius(radius int) Kernel {
	sideLength := radius*2 + 1

	return Kernel{
		radius:     radius,
		sideLength: sideLength,
		weights:    make([]kernelWeight, sideLength*sideLength),
	}
}

type kernelClip struct {
	Left   int
	Right  int
	Top    int
	Bottom int
}

type kernelWeight struct {
	R int32
	G int32
	B int32
	A int32
}

func (kw *kernelWeight) toNRGBA() color.NRGBA {
	return color.NRGBA{
		R: clip255(kw.R),
		G: clip255(kw.G),
		B: clip255(kw.B),
		A: clip255(kw.A),
	}
}

func clip255(v int32) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}
