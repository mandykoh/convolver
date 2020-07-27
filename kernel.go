package convolver

import (
	"fmt"
	"github.com/mandykoh/prism/srgb"
	"image"
	"image/color"
	"image/draw"
	"sync"
)

type opFunc func(img *image.NRGBA, x, y int) color.NRGBA

type Kernel struct {
	radius     int
	sideLength int
	weights    []kernelWeight
}

func (k *Kernel) ApplyMax(img image.Image, parallelism int) *image.NRGBA {
	return k.apply(convertImageToNRGBA(img), k.Max, parallelism)
}

func (k *Kernel) ApplyMin(img image.Image, parallelism int) *image.NRGBA {
	return k.apply(convertImageToNRGBA(img), k.Min, parallelism)
}

func (k *Kernel) ApplyAvg(img image.Image, parallelism int) *image.NRGBA {
	return k.apply(convertImageToNRGBA(img), k.Avg, parallelism)
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
			totalWeight.R += weight.R
			totalWeight.G += weight.G
			totalWeight.B += weight.B
			totalWeight.A += weight.A

			c := img.NRGBAAt(x+t-k.radius, y+s-k.radius)
			sum.R += srgb.From8Bit(c.R) * weight.R
			sum.G += srgb.From8Bit(c.G) * weight.G
			sum.B += srgb.From8Bit(c.B) * weight.B
			sum.A += srgb.From8Bit(c.A) * weight.A
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

			c := img.NRGBAAt(x+t-k.radius, y+s-k.radius)
			if v := srgb.From8Bit(c.R); v*weight.R > max.R*weight.R {
				max.R = v
			}
			if v := srgb.From8Bit(c.G); v*weight.G > max.G*weight.G {
				max.G = v
			}
			if v := srgb.From8Bit(c.B); v*weight.B > max.B*weight.B {
				max.B = v
			}
			if v := srgb.From8Bit(c.A); v*weight.A > max.A*weight.A {
				max.A = v
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

			c := img.NRGBAAt(x+t-k.radius, y+s-k.radius)
			if v := srgb.From8Bit(c.R); v*weight.R < min.R*weight.R {
				min.R = v
			}
			if v := srgb.From8Bit(c.G); v*weight.G < min.G*weight.G {
				min.G = v
			}
			if v := srgb.From8Bit(c.B); v*weight.B < min.B*weight.B {
				min.B = v
			}
			if v := srgb.From8Bit(c.A); v*weight.A < min.A*weight.A {
				min.A = v
			}
		}
	}

	return min.toNRGBA()
}

func (k *Kernel) SetWeightRGBA(x, y int, r, g, b, a float64) {
	k.weights[y*k.sideLength+x] = kernelWeight{
		R: r,
		G: g,
		B: b,
		A: a,
	}
}

func (k *Kernel) SetWeightUniform(x, y int, weight float64) {
	k.SetWeightRGBA(x, y, weight, weight, weight, weight)
}

func (k *Kernel) SetWeightsRGBA(weights [][4]float64) {
	if expectedWeights := k.sideLength * k.sideLength; expectedWeights != len(weights) {
		panic(fmt.Sprintf("kernel of radius %d requires exactly %d weights but %d provided", k.radius, expectedWeights, len(weights)))
	}

	for i := 0; i < len(weights); i++ {
		w := weights[i]
		k.weights[i] = kernelWeight{R: w[0], G: w[1], B: w[2], A: w[3]}
	}
}

func (k *Kernel) SetWeightsUniform(weights []float64) {
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
	R float64
	G float64
	B float64
	A float64
}

func (kw *kernelWeight) toNRGBA() color.NRGBA {
	return color.NRGBA{
		R: srgb.To8Bit(kw.R),
		G: srgb.To8Bit(kw.G),
		B: srgb.To8Bit(kw.B),
		A: srgb.To8Bit(kw.A),
	}
}

func convertImageToNRGBA(img image.Image) *image.NRGBA {
	inputImg, ok := img.(*image.NRGBA)
	if !ok {
		bounds := img.Bounds()
		inputImg = image.NewNRGBA(bounds)
		draw.Draw(inputImg, bounds, img, bounds.Min, draw.Src)
	}

	return inputImg
}
