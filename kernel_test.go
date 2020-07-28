package convolver

import (
	"fmt"
	"github.com/mandykoh/prism/srgb"
	"image"
	"image/color"
	"math/rand"
	"runtime"
	"sync"
	"testing"
)

func BenchmarkAggregation(b *testing.B) {
	inputImg := randomImage(4096, 4096)

	weights := []float32{
		0, 1, 0, 1, 0,
		1, 0, 1, 0, 1,
		0, 1, 0, 1, 0,
		1, 0, 1, 0, 1,
		0, 1, 0, 1, 0,
	}

	kernel := KernelWithRadius(2)
	kernel.SetWeightsUniform(weights)

	cases := []struct {
		OpName string
		Op     opFunc
	}{
		{OpName: "Avg", Op: kernel.Avg},
		{OpName: "Max", Op: kernel.Max},
		{OpName: "Min", Op: kernel.Min},
	}

	for _, c := range cases {
		b.Run(fmt.Sprintf("with %s operation", c.OpName), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = kernel.apply(inputImg, c.Op, runtime.NumCPU())
			}
		})
	}
}

func BenchmarkParallelisation(b *testing.B) {
	inputImg := randomImage(8192, 8192)

	// Gaussian blur kernel
	weights := []float32{
		1, 4, 6, 4, 1,
		4, 16, 24, 16, 4,
		6, 24, 36, 24, 6,
		4, 16, 24, 16, 4,
		1, 4, 6, 4, 1,
	}

	kernel := KernelWithRadius(2)
	kernel.SetWeightsUniform(weights)

	for threadCount := 1; threadCount <= runtime.NumCPU(); threadCount++ {
		b.Run(fmt.Sprintf("with parallelism %d", threadCount), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = kernel.ApplyAvg(inputImg, threadCount)
			}
		})
	}
}

func TestKernel(t *testing.T) {

	t.Run("apply()", func(t *testing.T) {
		img := randomImage(256, 256)

		mutex := sync.Mutex{}
		pixelsVisited := make(map[image.Point]struct{})

		op := func(_ *image.NRGBA, x, y int) color.NRGBA {
			mutex.Lock()
			defer mutex.Unlock()

			pixelsVisited[image.Pt(x, y)] = struct{}{}
			return img.NRGBAAt(x, y)
		}

		kernel := KernelWithRadius(0)
		result := kernel.apply(img, op, runtime.NumCPU())

		if expected, actual := img.Rect, result.Rect; expected.Dx() != actual.Dx() || expected.Dy() != actual.Dy() {
			t.Errorf("Expected resulting image to be %dx%d but was %dx%d", expected.Dx(), expected.Dy(), actual.Dx(), actual.Dy())
		}

		differentPixelCount := 0
		pixelsNotVisited := 0

		for i := img.Rect.Min.Y; i < img.Rect.Max.Y; i++ {
			for j := img.Rect.Min.X; j < img.Rect.Max.X; j++ {
				if expected, actual := img.NRGBAAt(j, i), result.NRGBAAt(j, i); expected != actual {
					differentPixelCount++
				}
				if _, ok := pixelsVisited[image.Pt(j, i)]; !ok {
					pixelsNotVisited++
				}
			}
		}

		if differentPixelCount > 0 {
			t.Errorf("Expected resulting image and input image to match but they differ at %d pixels", differentPixelCount)
		}
		if pixelsNotVisited > 0 {
			t.Errorf("Expected kernel operation to have been applied to all pixels but %d were not visited", pixelsNotVisited)
		}
	})

	t.Run("Avg()", func(t *testing.T) {
		img := randomImage(3, 3)

		t.Run("with uniform weights", func(t *testing.T) {
			expectedAvg := [4]float32{}
			for i := img.Rect.Min.Y; i < img.Rect.Max.Y; i++ {
				for j := img.Rect.Min.X; j < img.Rect.Max.X; j++ {
					c := srgb.ColorFromNRGBA(img.NRGBAAt(j, i))
					expectedAvg[0] += c.R
					expectedAvg[1] += c.G
					expectedAvg[2] += c.B
					expectedAvg[3] += c.A
				}
			}
			expectedAvg[0] /= float32(img.Rect.Dx() * img.Rect.Dy())
			expectedAvg[1] /= float32(img.Rect.Dx() * img.Rect.Dy())
			expectedAvg[2] /= float32(img.Rect.Dx() * img.Rect.Dy())
			expectedAvg[3] /= float32(img.Rect.Dx() * img.Rect.Dy())

			checkExpectedAvg := func(t *testing.T, kernel Kernel) {
				t.Helper()

				result := kernel.Avg(img, 1, 1)
				expectedColour := srgb.Color{
					R: expectedAvg[0],
					G: expectedAvg[1],
					B: expectedAvg[2],
					A: expectedAvg[3],
				}.To8Bit()

				if expected, actual := expectedColour, result; expected != actual {
					t.Errorf("Expected average to be %+v but was %+v", expected, actual)
				}
			}

			t.Run("includes all pixels covered by kernel", func(t *testing.T) {
				kernel := KernelWithRadius(1)
				for i := 0; i < kernel.SideLength(); i++ {
					for j := 0; j < kernel.SideLength(); j++ {
						kernel.SetWeightUniform(j, i, 1)
					}
				}

				checkExpectedAvg(t, kernel)
			})

			t.Run("clips kernel against edges of image", func(t *testing.T) {
				kernel := KernelWithRadius(2)
				for i := 0; i < kernel.SideLength(); i++ {
					for j := 0; j < kernel.SideLength(); j++ {
						kernel.SetWeightUniform(j, i, 1)
					}
				}

				checkExpectedAvg(t, kernel)
			})
		})

		t.Run("scales pixel values by kernel weights", func(t *testing.T) {
			totalWeight := float32(0.0)
			kernel := KernelWithRadius(1)
			for i := 0; i < kernel.SideLength(); i++ {
				for j := 0; j < kernel.SideLength(); j++ {
					weight := float32(i + j)
					totalWeight += weight
					kernel.SetWeightUniform(j, i, weight)
				}
			}

			avg := [4]float32{}
			for row, i := float32(0), img.Rect.Min.Y; i < img.Rect.Max.Y; row, i = row+1, i+1 {
				for col, j := float32(0), img.Rect.Min.X; j < img.Rect.Max.X; col, j = col+1, j+1 {
					c := srgb.ColorFromNRGBA(img.NRGBAAt(j, i))
					avg[0] += c.R * (row + col)
					avg[1] += c.G * (row + col)
					avg[2] += c.B * (row + col)
					avg[3] += c.A * (row + col)
				}
			}
			avg[0] /= totalWeight
			avg[1] /= totalWeight
			avg[2] /= totalWeight
			avg[3] /= totalWeight

			result := kernel.Avg(img, 1, 1)
			expectedColour := srgb.Color{R: avg[0], G: avg[1], B: avg[2], A: avg[3]}.To8Bit()

			if expected, actual := expectedColour, result; expected != actual {
				t.Errorf("Expected average to be %+v but was %+v", expected, actual)
			}
		})
	})

	t.Run("clipToBounds()", func(t *testing.T) {

		// 5x5 image with origin at 10,10
		bounds := image.Rect(10, 10, 15, 15)

		// 3x3 kernel
		kernel := KernelWithRadius(1)

		cases := []struct {
			Side         string
			KernelCentre image.Point
			ExpectedClip kernelClip
		}{
			{
				Side:         "centre",
				KernelCentre: image.Pt(12, 12),
				ExpectedClip: kernelClip{0, 0, 0, 0},
			},
			{
				Side:         "left inside",
				KernelCentre: image.Pt(10, 12),
				ExpectedClip: kernelClip{1, 0, 0, 0},
			},
			{
				Side:         "left outside",
				KernelCentre: image.Pt(9, 12),
				ExpectedClip: kernelClip{2, 0, 0, 0},
			},
			{
				Side:         "right inside",
				KernelCentre: image.Pt(14, 11),
				ExpectedClip: kernelClip{0, 1, 0, 0},
			},
			{
				Side:         "right outside",
				KernelCentre: image.Pt(15, 11),
				ExpectedClip: kernelClip{0, 2, 0, 0},
			},
			{
				Side:         "top inside",
				KernelCentre: image.Pt(12, 10),
				ExpectedClip: kernelClip{0, 0, 1, 0},
			},
			{
				Side:         "top outside",
				KernelCentre: image.Pt(12, 9),
				ExpectedClip: kernelClip{0, 0, 2, 0},
			},
			{
				Side:         "bottom inside",
				KernelCentre: image.Pt(12, 14),
				ExpectedClip: kernelClip{0, 0, 0, 1},
			},
			{
				Side:         "bottom outside",
				KernelCentre: image.Pt(12, 15),
				ExpectedClip: kernelClip{0, 0, 0, 2},
			},
			{
				Side:         "top left inside",
				KernelCentre: image.Pt(10, 10),
				ExpectedClip: kernelClip{1, 0, 1, 0},
			},
			{
				Side:         "top left outside",
				KernelCentre: image.Pt(9, 9),
				ExpectedClip: kernelClip{2, 0, 2, 0},
			},
			{
				Side:         "bottom right inside",
				KernelCentre: image.Pt(14, 14),
				ExpectedClip: kernelClip{0, 1, 0, 1},
			},
			{
				Side:         "bottom right outside",
				KernelCentre: image.Pt(15, 15),
				ExpectedClip: kernelClip{0, 2, 0, 2},
			},
		}

		for _, c := range cases {
			clip := kernel.clipToBounds(bounds, c.KernelCentre.X, c.KernelCentre.Y)
			if expected, actual := c.ExpectedClip, clip; expected != actual {
				t.Errorf("Expected clip at %s of bounds to be %+v but was %+v", c.Side, expected, actual)
			}
		}
	})

	t.Run("Max()", func(t *testing.T) {
		img := randomImage(3, 3)

		t.Run("with uniform weights", func(t *testing.T) {

			checkExpectedMax := func(t *testing.T, kernel Kernel) {
				t.Helper()

				max := [4]float32{}

				for i := img.Rect.Min.Y; i < img.Rect.Max.Y; i++ {
					for j := img.Rect.Min.X; j < img.Rect.Max.X; j++ {
						c := srgb.ColorFromNRGBA(img.NRGBAAt(j, i))

						if c.R > max[0] {
							max[0] = c.R
						}
						if c.G > max[1] {
							max[1] = c.G
						}
						if c.B > max[2] {
							max[2] = c.B
						}
						if c.A > max[3] {
							max[3] = c.A
						}
					}
				}

				result := kernel.Max(img, 1, 1)
				expectedColour := srgb.Color{R: max[0], G: max[1], B: max[2], A: max[3]}.To8Bit()

				if expected, actual := expectedColour, result; expected != actual {
					t.Errorf("Expected max to be %+v but was %+v", expected, actual)
				}
			}

			t.Run("includes all pixels covered by kernel", func(t *testing.T) {
				kernel := KernelWithRadius(1)
				for i := 0; i < kernel.SideLength(); i++ {
					for j := 0; j < kernel.SideLength(); j++ {
						kernel.SetWeightUniform(j, i, 1)
					}
				}

				checkExpectedMax(t, kernel)
			})

			t.Run("clips kernel against edges of image", func(t *testing.T) {
				weights := []float32{
					-1, -1, -1, -1, -1,
					-1, 1, 1, 1, -1,
					-1, 1, 1, 1, -1,
					-1, 1, 1, 1, -1,
					-1, -1, -1, -1, -1,
				}

				kernel := KernelWithRadius(2)
				kernel.SetWeightsUniform(weights)

				checkExpectedMax(t, kernel)
			})
		})

		t.Run("ignores pixel values with zero weight", func(t *testing.T) {
			weights := []float32{
				0, 1, 0,
				1, 0, 1,
				0, 1, 0,
			}

			kernel := KernelWithRadius(1)
			kernel.SetWeightsUniform(weights)

			max := [4]float32{}

			for row, i := 0, img.Rect.Min.Y; i < img.Rect.Max.Y; row, i = row+1, i+1 {
				for col, j := 0, img.Rect.Min.X; j < img.Rect.Max.X; col, j = col+1, j+1 {
					w := weights[row*kernel.SideLength()+col]
					if w == 0 {
						continue
					}

					c := srgb.ColorFromNRGBA(img.NRGBAAt(j, i))
					if c.R > max[0] {
						max[0] = c.R
					}
					if c.G > max[1] {
						max[1] = c.G
					}
					if c.B > max[2] {
						max[2] = c.B
					}
					if c.A > max[3] {
						max[3] = c.A
					}
				}
			}

			result := kernel.Max(img, 1, 1)
			expectedColour := srgb.Color{R: max[0], G: max[1], B: max[2], A: max[3]}.To8Bit()

			if expected, actual := expectedColour, result; expected != actual {
				t.Errorf("Expected max to be %+v but was %+v", expected, actual)
			}
		})
	})

	t.Run("Min()", func(t *testing.T) {
		img := randomImage(3, 3)

		t.Run("with uniform weights", func(t *testing.T) {

			checkExpectedMin := func(t *testing.T, kernel Kernel) {
				t.Helper()

				min := [4]float32{255, 255, 255, 255}

				for i := img.Rect.Min.Y; i < img.Rect.Max.Y; i++ {
					for j := img.Rect.Min.X; j < img.Rect.Max.X; j++ {
						c := srgb.ColorFromNRGBA(img.NRGBAAt(j, i))

						if c.R < min[0] {
							min[0] = c.R
						}
						if c.G < min[1] {
							min[1] = c.G
						}
						if c.B < min[2] {
							min[2] = c.B
						}
						if c.A < min[3] {
							min[3] = c.A
						}
					}
				}

				result := kernel.Min(img, 1, 1)
				expectedColour := srgb.Color{R: min[0], G: min[1], B: min[2], A: min[3]}.To8Bit()

				if expected, actual := expectedColour, result; expected != actual {
					t.Errorf("Expected min to be %+v but was %+v", expected, actual)
				}
			}

			t.Run("includes all pixels covered by kernel", func(t *testing.T) {
				kernel := KernelWithRadius(1)
				for i := 0; i < kernel.SideLength(); i++ {
					for j := 0; j < kernel.SideLength(); j++ {
						kernel.SetWeightUniform(j, i, 1)
					}
				}

				checkExpectedMin(t, kernel)
			})

			t.Run("clips kernel against edges of image", func(t *testing.T) {
				kernel := KernelWithRadius(2)
				for i := 0; i < kernel.SideLength(); i++ {
					for j := 0; j < kernel.SideLength(); j++ {
						kernel.SetWeightUniform(j, i, 1)
					}
				}

				checkExpectedMin(t, kernel)
			})
		})

		t.Run("ignores pixel values with zero weight", func(t *testing.T) {
			weights := []float32{
				0, 1, 0,
				1, 0, 1,
				0, 1, 0,
			}

			kernel := KernelWithRadius(1)
			kernel.SetWeightsUniform(weights)

			min := [4]float32{255, 255, 255, 255}

			for row, i := 0, img.Rect.Min.Y; i < img.Rect.Max.Y; row, i = row+1, i+1 {
				for col, j := 0, img.Rect.Min.X; j < img.Rect.Max.X; col, j = col+1, j+1 {
					w := weights[row*kernel.SideLength()+col]
					if w == 0 {
						continue
					}

					c := srgb.ColorFromNRGBA(img.NRGBAAt(j, i))
					if c.R < min[0] {
						min[0] = c.R
					}
					if c.G < min[1] {
						min[1] = c.G
					}
					if c.B < min[2] {
						min[2] = c.B
					}
					if c.A < min[3] {
						min[3] = c.A
					}
				}
			}

			result := kernel.Min(img, 1, 1)
			expectedColour := srgb.Color{R: min[0], G: min[1], B: min[2], A: min[3]}.To8Bit()

			if expected, actual := expectedColour, result; expected != actual {
				t.Errorf("Expected min to be %+v but was %+v", expected, actual)
			}
		})
	})
}

func randomImage(w, h int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))

	for i := img.Rect.Min.Y; i < img.Rect.Max.Y; i++ {
		for j := img.Rect.Min.X; j < img.Rect.Max.X; j++ {
			img.SetNRGBA(j, i, color.NRGBA{
				R: uint8(rand.Intn(256)),
				G: uint8(rand.Intn(256)),
				B: uint8(rand.Intn(256)),
				A: uint8(rand.Intn(256)),
			})
		}
	}

	return img
}
