package convolver

import (
	"fmt"
	"image"
	"image/color"
	"sync"

	"github.com/mandykoh/prism"
	"github.com/mandykoh/prism/srgb"
)

type opFunc func(img *image.NRGBA, x, y int) color.NRGBA

type Kernel struct {
	radius     int
	sideLength int
	weights    []kernelWeight
}

func (k *Kernel) ApplyMax(img image.Image, parallelism int) *image.NRGBA {
	return k.apply(prism.ConvertImageToNRGBA(img), k.Max, parallelism)
}

func (k *Kernel) ApplyMin(img image.Image, parallelism int) *image.NRGBA {
	return k.apply(prism.ConvertImageToNRGBA(img), k.Min, parallelism)
}

func (k *Kernel) ApplyAvg(img image.Image, parallelism int) *image.NRGBA {
	return k.apply(prism.ConvertImageToNRGBA(img), k.Avg, parallelism)
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

func (k *Kernel) Avg(img *image.NRGBA, x, y int) color.NRGBA {
	clip := k.clipToBounds(img.Rect, x, y)

	totalWeight := kernelWeight{}
	sum := kernelWeight{}

	for s := clip.Top; s < k.sideLength-clip.Bottom; s++ {
		for t := clip.Left; t < k.sideLength-clip.Right; t++ {
			weight := k.weights[s*k.sideLength+t]
			// totalWeight = totalWeight + weight
			totalWeight = totalWeight.add(weight)

			c, a := srgb.ColorFromNRGBA(img.NRGBAAt(x+t-k.radius, y+s-k.radius))
			// sum = sum + (weight * c)
			sum = sum.add(weight.mul(kernelWeight{c.R, c.G, c.B, a}))
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

func (k *Kernel) clipToBounds(bounds image.Rectangle, x, y int) kernelClip {
	clip := kernelClip{}

	if edgeDist := x - bounds.Min.X; edgeDist < k.radius {
		clip.Left = k.radius - edgeDist
	}
	if edgeDist := bounds.Max.X - x - 1; edgeDist < k.radius {
		clip.Right = k.radius - edgeDist
	}
	if edgeDist := y - bounds.Min.Y; edgeDist < k.radius {
		clip.Top = k.radius - edgeDist
	}
	if edgeDist := bounds.Max.Y - y - 1; edgeDist < k.radius {
		clip.Bottom = k.radius - edgeDist
	}

	return clip
}

func (k *Kernel) Max(img *image.NRGBA, x, y int) color.NRGBA {
	clip := k.clipToBounds(img.Rect, x, y)

	max := kernelWeight{}

	for s := clip.Top; s < k.sideLength-clip.Bottom; s++ {
		for t := clip.Left; t < k.sideLength-clip.Right; t++ {
			weight := k.weights[s*k.sideLength+t]

			c, a := srgb.ColorFromNRGBA(img.NRGBAAt(x+t-k.radius, y+s-k.radius))
			multiplication := weight.mul(kernelWeight{c.R, c.G, c.B, a})
			maximum := multiplication.max(max)

			if weight.R != 0 {
				max.R = maximum.R
			}
			if weight.G != 0 {
				max.G = maximum.G
			}
			if weight.B != 0 {
				max.B = maximum.B
			}
			if weight.A != 0 {
				max.A = maximum.A
			}
		}
	}

	return max.toNRGBA()
}

func (k *Kernel) Min(img *image.NRGBA, x, y int) color.NRGBA {
	clip := k.clipToBounds(img.Rect, x, y)

	min := kernelWeight{255, 255, 255, 255}

	for s := clip.Top; s < k.sideLength-clip.Bottom; s++ {
		for t := clip.Left; t < k.sideLength-clip.Right; t++ {
			weight := k.weights[s*k.sideLength+t]

			c, a := srgb.ColorFromNRGBA(img.NRGBAAt(x+t-k.radius, y+s-k.radius))
			multiplication := weight.mul(kernelWeight{c.R, c.G, c.B, a})
			minimum := multiplication.min(min)

			if weight.R != 0 {
				min.R = minimum.R
			}
			if weight.G != 0 {
				min.G = minimum.G
			}
			if weight.B != 0 {
				min.B = minimum.B
			}
			if weight.A != 0 {
				min.A = minimum.A
			}
		}
	}

	return min.toNRGBA()
}

func (k *Kernel) SetWeightRGBA(x, y int, r, g, b, a float32) {
	k.weights[y*k.sideLength+x] = kernelWeight{R: r, G: g, B: b, A: a}
}

func (k *Kernel) SetWeightUniform(x, y int, weight float32) {
	k.SetWeightRGBA(x, y, weight, weight, weight, weight)
}

func (k *Kernel) SetWeightsRGBA(weights [][4]float32) {
	if expectedWeights := k.sideLength * k.sideLength; expectedWeights != len(weights) {
		panic(fmt.Sprintf("kernel of radius %d requires exactly %d weights but %d provided", k.radius, expectedWeights, len(weights)))
	}

	for i := 0; i < len(weights); i++ {
		w := weights[i]
		k.weights[i] = kernelWeight{R: w[0], G: w[1], B: w[2], A: w[3]}
	}
}

func (k *Kernel) SetWeightsUniform(weights []float32) {
	if expectedWeights := k.sideLength * k.sideLength; expectedWeights != len(weights) {
		panic(fmt.Sprintf("kernel of radius %d requires exactly %d weights but %d provided", k.radius, expectedWeights, len(weights)))
	}

	for i := 0; i < len(weights); i++ {
		w := weights[i]
		k.weights[i] = kernelWeight{R: w, G: w, B: w, A: w}
	}
}

func (k *Kernel) SideLength() int {
	return k.sideLength
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
	R float32
	G float32
	B float32
	A float32
}

func (kw kernelWeight) add(b kernelWeight) (r kernelWeight) {
	add(&kw, &b, &r)
	return
}

func (kw kernelWeight) mul(b kernelWeight) (r kernelWeight) {
	mul(&kw, &b, &r)
	return
}

func (kw kernelWeight) max(b kernelWeight) (r kernelWeight) {
	max(&kw, &b, &r)
	return
}

func (kw kernelWeight) min(b kernelWeight) (r kernelWeight) {
	min(&kw, &b, &r)
	return
}

func (kw *kernelWeight) toNRGBA() color.NRGBA {
	return srgb.ColorFromLinear(kw.R, kw.G, kw.B).ToNRGBA(kw.A)
}
