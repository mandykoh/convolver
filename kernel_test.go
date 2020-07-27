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

	weights := []float64{
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
	weights := []float64{
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
			expectedAvg := [4]float64{}
			for i := img.Rect.Min.Y; i < img.Rect.Max.Y; i++ {
				for j := img.Rect.Min.X; j < img.Rect.Max.X; j++ {
					c := img.NRGBAAt(j, i)
					expectedAvg[0] += srgb.From8Bit(c.R)
					expectedAvg[1] += srgb.From8Bit(c.G)
					expectedAvg[2] += srgb.From8Bit(c.B)
					expectedAvg[3] += srgb.From8Bit(c.A)
				}
			}
			expectedAvg[0] /= float64(img.Rect.Dx() * img.Rect.Dy())
			expectedAvg[1] /= float64(img.Rect.Dx() * img.Rect.Dy())
			expectedAvg[2] /= float64(img.Rect.Dx() * img.Rect.Dy())
			expectedAvg[3] /= float64(img.Rect.Dx() * img.Rect.Dy())

			checkExpectedAvg := func(t *testing.T, kernel Kernel) {
				t.Helper()

				result := kernel.Avg(img, 1, 1)

				if expected, actual := srgb.To8Bit(expectedAvg[0]), result.R; expected != actual {
					t.Errorf("Expected average of red channel to be %d but was %d", expected, actual)
				}
				if expected, actual := srgb.To8Bit(expectedAvg[1]), result.G; expected != actual {
					t.Errorf("Expected average of green channel to be %d but was %d", expected, actual)
				}
				if expected, actual := srgb.To8Bit(expectedAvg[2]), result.B; expected != actual {
					t.Errorf("Expected average of blue channel to be %d but was %d", expected, actual)
				}
				if expected, actual := srgb.To8Bit(expectedAvg[3]), result.A; expected != actual {
					t.Errorf("Expected average of alpha channel to be %d but was %d", expected, actual)
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
			totalWeight := 0.0
			kernel := KernelWithRadius(1)
			for i := 0; i < kernel.SideLength(); i++ {
				for j := 0; j < kernel.SideLength(); j++ {
					weight := float64(i + j)
					totalWeight += weight
					kernel.SetWeightUniform(j, i, weight)
				}
			}

			avg := [4]float64{}
			for row, i := 0.0, img.Rect.Min.Y; i < img.Rect.Max.Y; row, i = row+1, i+1 {
				for col, j := 0.0, img.Rect.Min.X; j < img.Rect.Max.X; col, j = col+1, j+1 {
					c := img.NRGBAAt(j, i)
					avg[0] += srgb.From8Bit(c.R) * (row + col)
					avg[1] += srgb.From8Bit(c.G) * (row + col)
					avg[2] += srgb.From8Bit(c.B) * (row + col)
					avg[3] += srgb.From8Bit(c.A) * (row + col)
				}
			}
			avg[0] /= totalWeight
			avg[1] /= totalWeight
			avg[2] /= totalWeight
			avg[3] /= totalWeight

			result := kernel.Avg(img, 1, 1)

			if expected, actual := srgb.To8Bit(avg[0]), result.R; expected != actual {
				t.Errorf("Expected average of red channel to be %d but was %d", expected, actual)
			}
			if expected, actual := srgb.To8Bit(avg[1]), result.G; expected != actual {
				t.Errorf("Expected average of green channel to be %d but was %d", expected, actual)
			}
			if expected, actual := srgb.To8Bit(avg[2]), result.B; expected != actual {
				t.Errorf("Expected average of blue channel to be %d but was %d", expected, actual)
			}
			if expected, actual := srgb.To8Bit(avg[3]), result.A; expected != actual {
				t.Errorf("Expected average of alpha channel to be %d but was %d", expected, actual)
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

				expectedMax := [4]float64{}

				for i := img.Rect.Min.Y; i < img.Rect.Max.Y; i++ {
					for j := img.Rect.Min.X; j < img.Rect.Max.X; j++ {
						c := img.NRGBAAt(j, i)

						if v := srgb.From8Bit(c.R); v > expectedMax[0] {
							expectedMax[0] = v
						}
						if v := srgb.From8Bit(c.G); v > expectedMax[1] {
							expectedMax[1] = v
						}
						if v := srgb.From8Bit(c.B); v > expectedMax[2] {
							expectedMax[2] = v
						}
						if v := srgb.From8Bit(c.A); v > expectedMax[3] {
							expectedMax[3] = v
						}
					}
				}

				result := kernel.Max(img, 1, 1)

				if expected, actual := srgb.To8Bit(expectedMax[0]), result.R; expected != actual {
					t.Errorf("Expected max of red channel to be %d but was %d", expected, actual)
				}
				if expected, actual := srgb.To8Bit(expectedMax[1]), result.G; expected != actual {
					t.Errorf("Expected max of green channel to be %d but was %d", expected, actual)
				}
				if expected, actual := srgb.To8Bit(expectedMax[2]), result.B; expected != actual {
					t.Errorf("Expected max of blue channel to be %d but was %d", expected, actual)
				}
				if expected, actual := srgb.To8Bit(expectedMax[3]), result.A; expected != actual {
					t.Errorf("Expected max of alpha channel to be %d but was %d", expected, actual)
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
				weights := []float64{
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
			weights := []float64{
				0, 1, 0,
				1, 0, 1,
				0, 1, 0,
			}

			kernel := KernelWithRadius(1)
			kernel.SetWeightsUniform(weights)

			max := [4]float64{}

			for row, i := 0, img.Rect.Min.Y; i < img.Rect.Max.Y; row, i = row+1, i+1 {
				for col, j := 0, img.Rect.Min.X; j < img.Rect.Max.X; col, j = col+1, j+1 {
					w := weights[row*kernel.SideLength()+col]
					if w == 0 {
						continue
					}

					c := img.NRGBAAt(j, i)
					if v := srgb.From8Bit(c.R); v > max[0] {
						max[0] = v
					}
					if v := srgb.From8Bit(c.G); v > max[1] {
						max[1] = v
					}
					if v := srgb.From8Bit(c.B); v > max[2] {
						max[2] = v
					}
					if v := srgb.From8Bit(c.A); v > max[3] {
						max[3] = v
					}
				}
			}

			result := kernel.Max(img, 1, 1)

			if expected, actual := srgb.To8Bit(max[0]), result.R; expected != actual {
				t.Errorf("Expected max of red channel to be %d but was %d", expected, actual)
			}
			if expected, actual := srgb.To8Bit(max[1]), result.G; expected != actual {
				t.Errorf("Expected max of green channel to be %d but was %d", expected, actual)
			}
			if expected, actual := srgb.To8Bit(max[2]), result.B; expected != actual {
				t.Errorf("Expected max of blue channel to be %d but was %d", expected, actual)
			}
			if expected, actual := srgb.To8Bit(max[3]), result.A; expected != actual {
				t.Errorf("Expected max of alpha channel to be %d but was %d", expected, actual)
			}
		})
	})

	t.Run("Min()", func(t *testing.T) {
		img := randomImage(3, 3)

		t.Run("with uniform weights", func(t *testing.T) {

			checkExpectedMin := func(t *testing.T, kernel Kernel) {
				t.Helper()

				expectedMin := [4]float64{255, 255, 255, 255}

				for i := img.Rect.Min.Y; i < img.Rect.Max.Y; i++ {
					for j := img.Rect.Min.X; j < img.Rect.Max.X; j++ {
						c := img.NRGBAAt(j, i)

						if v := srgb.From8Bit(c.R); v < expectedMin[0] {
							expectedMin[0] = v
						}
						if v := srgb.From8Bit(c.G); v < expectedMin[1] {
							expectedMin[1] = v
						}
						if v := srgb.From8Bit(c.B); v < expectedMin[2] {
							expectedMin[2] = v
						}
						if v := srgb.From8Bit(c.A); v < expectedMin[3] {
							expectedMin[3] = v
						}
					}
				}

				result := kernel.Min(img, 1, 1)

				if expected, actual := srgb.To8Bit(expectedMin[0]), result.R; expected != actual {
					t.Errorf("Expected min of red channel to be %d but was %d", expected, actual)
				}
				if expected, actual := srgb.To8Bit(expectedMin[1]), result.G; expected != actual {
					t.Errorf("Expected min of green channel to be %d but was %d", expected, actual)
				}
				if expected, actual := srgb.To8Bit(expectedMin[2]), result.B; expected != actual {
					t.Errorf("Expected min of blue channel to be %d but was %d", expected, actual)
				}
				if expected, actual := srgb.To8Bit(expectedMin[3]), result.A; expected != actual {
					t.Errorf("Expected min of alpha channel to be %d but was %d", expected, actual)
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
			weights := []float64{
				0, 1, 0,
				1, 0, 1,
				0, 1, 0,
			}

			kernel := KernelWithRadius(1)
			kernel.SetWeightsUniform(weights)

			min := [4]float64{255, 255, 255, 255}

			for row, i := 0, img.Rect.Min.Y; i < img.Rect.Max.Y; row, i = row+1, i+1 {
				for col, j := 0, img.Rect.Min.X; j < img.Rect.Max.X; col, j = col+1, j+1 {
					w := weights[row*kernel.SideLength()+col]
					if w == 0 {
						continue
					}

					c := img.NRGBAAt(j, i)
					if v := srgb.From8Bit(c.R); v < min[0] {
						min[0] = v
					}
					if v := srgb.From8Bit(c.G); v < min[1] {
						min[1] = v
					}
					if v := srgb.From8Bit(c.B); v < min[2] {
						min[2] = v
					}
					if v := srgb.From8Bit(c.A); v < min[3] {
						min[3] = v
					}
				}
			}

			result := kernel.Min(img, 1, 1)

			if expected, actual := srgb.To8Bit(min[0]), result.R; expected != actual {
				t.Errorf("Expected min of red channel to be %d but was %d", expected, actual)
			}
			if expected, actual := srgb.To8Bit(min[1]), result.G; expected != actual {
				t.Errorf("Expected min of green channel to be %d but was %d", expected, actual)
			}
			if expected, actual := srgb.To8Bit(min[2]), result.B; expected != actual {
				t.Errorf("Expected min of blue channel to be %d but was %d", expected, actual)
			}
			if expected, actual := srgb.To8Bit(min[3]), result.A; expected != actual {
				t.Errorf("Expected min of alpha channel to be %d but was %d", expected, actual)
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
