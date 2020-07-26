package convolver

import (
	"fmt"
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
	return k.apply(convertToNRGBA(img), k.Max, parallelism)
}

func (k *Kernel) ApplyMin(img image.Image, parallelism int) *image.NRGBA {
	return k.apply(convertToNRGBA(img), k.Min, parallelism)
}

func (k *Kernel) ApplyAvg(img image.Image, parallelism int) *image.NRGBA {
	return k.apply(convertToNRGBA(img), k.Avg, parallelism)
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
	clip := k.clipToBounds(img.Rect, x, y)

	min := kernelWeight{255, 255, 255, 255}

	for s := clip.Top; s < k.sideLength-clip.Bottom; s++ {
		for t := clip.Left; t < k.sideLength-clip.Right; t++ {
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

func (k *Kernel) SetWeightRGBA(x, y int, r, g, b, a int32) {
	k.weights[y*k.sideLength+x] = kernelWeight{
		R: r,
		G: g,
		B: b,
		A: a,
	}
}

func (k *Kernel) SetWeightUniform(x, y int, weight int32) {
	k.SetWeightRGBA(x, y, weight, weight, weight, weight)
}

func (k *Kernel) SetWeightsRGBA(weights [][4]int32) {
	if expectedWeights := k.sideLength * k.sideLength; expectedWeights != len(weights) {
		panic(fmt.Sprintf("kernel of radius %d requires exactly %d weights but %d provided", k.radius, expectedWeights, len(weights)))
	}

	for i := 0; i < len(weights); i++ {
		w := weights[i]
		k.weights[i] = kernelWeight{R: w[0], G: w[1], B: w[2], A: w[3]}
	}
}

func (k *Kernel) SetWeightsUniform(weights []int32) {
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

func convertToNRGBA(img image.Image) *image.NRGBA {
	inputImg, ok := img.(*image.NRGBA)
	if !ok {
		bounds := img.Bounds()
		inputImg = image.NewNRGBA(bounds)
		draw.Draw(inputImg, bounds, img, bounds.Min, draw.Src)
	}

	return inputImg
}
