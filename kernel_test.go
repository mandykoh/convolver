package convolver

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"log"
	"math/rand"
	"os"
	"path"
	"runtime"
	"sync"
	"testing"
	"time"
)

func BenchmarkParallelisation(b *testing.B) {
	inputImg := randomImage(8192, 8192)

	// Gaussian blur kernel
	weights := []int32{
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

func ExampleKernel_channelExtraction() {
	imgFile, err := os.Open("test-images/avocado.png")
	if err != nil {
		log.Panicf("Error opening input image: %v", err)
	}
	defer imgFile.Close()

	img, err := png.Decode(imgFile)
	if err != nil {
		log.Panicf("Error decoding PNG: %v", err)
	}

	var inputImg *image.NRGBA
	inputImg, ok := img.(*image.NRGBA)
	if !ok {
		inputImg = image.NewNRGBA(img.Bounds())
		draw.Draw(inputImg, inputImg.Bounds(), img, image.Point{}, draw.Src)
	}

	kernel := KernelWithRadius(0)
	kernel.SetWeightRGBA(0, 0, 0, 0, 1, 1)

	startTime := time.Now()
	result := kernel.ApplyAvg(inputImg, runtime.NumCPU())
	endTime := time.Now()

	log.Printf("Channel extraction applied in %v", endTime.Sub(startTime))

	_ = os.Mkdir("example-output", os.ModePerm)
	outFilePath := path.Join("example-output", "example-channel-extraction.png")
	outFile, err := os.Create(outFilePath)
	if err != nil {
		log.Panicf("Error creating output file: %v", err)
	}
	defer outFile.Close()

	err = png.Encode(outFile, result)
	if err != nil {
		log.Panicf("Error encoding output image: %v", err)
	}

	err = outFile.Close()
	if err != nil {
		log.Panicf("Error closing output file: %v", err)
	}
	log.Printf("Output written to %s", outFilePath)

	// Output:
}

func ExampleKernel_gaussianBlur() {
	const numPasses = 8

	imgFile, err := os.Open("test-images/avocado.png")
	if err != nil {
		log.Panicf("Error opening input image: %v", err)
	}
	defer imgFile.Close()

	img, err := png.Decode(imgFile)
	if err != nil {
		log.Panicf("Error decoding PNG: %v", err)
	}

	var inputImg *image.NRGBA
	inputImg, ok := img.(*image.NRGBA)
	if !ok {
		inputImg = image.NewNRGBA(img.Bounds())
		draw.Draw(inputImg, inputImg.Bounds(), img, image.Point{}, draw.Src)
	}

	weights := []int32{
		1, 4, 6, 4, 1,
		4, 16, 24, 16, 4,
		6, 24, 36, 24, 6,
		4, 16, 24, 16, 4,
		1, 4, 6, 4, 1,
	}

	kernel := KernelWithRadius(2)
	kernel.SetWeightsUniform(weights)

	startTime := time.Now()
	result := inputImg
	for i := 0; i < numPasses; i++ {
		result = kernel.ApplyAvg(result, runtime.NumCPU())
	}
	endTime := time.Now()

	log.Printf("Gaussian blur applied in %v", endTime.Sub(startTime))

	_ = os.Mkdir("example-output", os.ModePerm)
	outFilePath := path.Join("example-output", "example-gaussian-blur.png")
	outFile, err := os.Create(outFilePath)
	if err != nil {
		log.Panicf("Error creating output file: %v", err)
	}
	defer outFile.Close()

	err = png.Encode(outFile, result)
	if err != nil {
		log.Panicf("Error encoding output image: %v", err)
	}

	err = outFile.Close()
	if err != nil {
		log.Panicf("Error closing output file: %v", err)
	}
	log.Printf("Output written to %s", outFilePath)

	// Output:
}

func ExampleKernel_sharpen() {
	imgFile, err := os.Open("test-images/avocado.png")
	if err != nil {
		log.Panicf("Error opening input image: %v", err)
	}
	defer imgFile.Close()

	img, err := png.Decode(imgFile)
	if err != nil {
		log.Panicf("Error decoding PNG: %v", err)
	}

	var inputImg *image.NRGBA
	inputImg, ok := img.(*image.NRGBA)
	if !ok {
		inputImg = image.NewNRGBA(img.Bounds())
		draw.Draw(inputImg, inputImg.Bounds(), img, image.Point{}, draw.Src)
	}

	weights := []int32{
		0, -1, 0,
		-1, 5, -1,
		0, -1, 0,
	}

	kernel := KernelWithRadius(1)
	kernel.SetWeightsUniform(weights)

	startTime := time.Now()
	result := kernel.ApplyAvg(inputImg, runtime.NumCPU())
	endTime := time.Now()

	log.Printf("Sharpen applied in %v", endTime.Sub(startTime))

	_ = os.Mkdir("example-output", os.ModePerm)
	outFilePath := path.Join("example-output", "example-sharpen.png")
	outFile, err := os.Create(outFilePath)
	if err != nil {
		log.Panicf("Error creating output file: %v", err)
	}
	defer outFile.Close()

	err = png.Encode(outFile, result)
	if err != nil {
		log.Panicf("Error encoding output image: %v", err)
	}

	err = outFile.Close()
	if err != nil {
		log.Panicf("Error closing output file: %v", err)
	}
	log.Printf("Output written to %s", outFilePath)

	// Output:
}

func ExampleKernel_dilateErode() {
	const numPasses = 5

	imgFile, err := os.Open("test-images/convolver-alpha-1024.png")
	if err != nil {
		log.Panicf("Error opening input image: %v", err)
	}
	defer imgFile.Close()

	img, err := png.Decode(imgFile)
	if err != nil {
		log.Panicf("Error decoding PNG: %v", err)
	}

	var inputImg *image.NRGBA
	inputImg, ok := img.(*image.NRGBA)
	if !ok {
		inputImg = image.NewNRGBA(img.Bounds())
		draw.Draw(inputImg, inputImg.Bounds(), img, image.Point{}, draw.Src)
	}

	weights := []int32{
		0, 1, 1, 1, 0,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		0, 1, 1, 1, 0,
	}

	kernel := KernelWithRadius(2)
	kernel.SetWeightsUniform(weights)

	startTime := time.Now()
	result := inputImg
	for i := 0; i < numPasses; i++ {
		result = kernel.ApplyMax(result, runtime.NumCPU())
	}
	for i := 0; i < numPasses; i++ {
		result = kernel.ApplyMin(result, runtime.NumCPU())
	}
	endTime := time.Now()

	log.Printf("Dilate-erode applied in %v", endTime.Sub(startTime))

	_ = os.Mkdir("example-output", os.ModePerm)
	outFilePath := path.Join("example-output", "example-dilate-erode.png")
	outFile, err := os.Create(outFilePath)
	if err != nil {
		log.Panicf("Error creating output file: %v", err)
	}
	defer outFile.Close()

	err = png.Encode(outFile, result)
	if err != nil {
		log.Panicf("Error encoding output image: %v", err)
	}

	err = outFile.Close()
	if err != nil {
		log.Panicf("Error closing output file: %v", err)
	}
	log.Printf("Output written to %s", outFilePath)

	// Output:
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
			expectedAvg := [4]int32{}
			for i := img.Rect.Min.Y; i < img.Rect.Max.Y; i++ {
				for j := img.Rect.Min.X; j < img.Rect.Max.X; j++ {
					c := img.NRGBAAt(j, i)
					expectedAvg[0] += int32(c.R)
					expectedAvg[1] += int32(c.G)
					expectedAvg[2] += int32(c.B)
					expectedAvg[3] += int32(c.A)
				}
			}
			expectedAvg[0] /= int32(img.Rect.Dx() * img.Rect.Dy())
			expectedAvg[1] /= int32(img.Rect.Dx() * img.Rect.Dy())
			expectedAvg[2] /= int32(img.Rect.Dx() * img.Rect.Dy())
			expectedAvg[3] /= int32(img.Rect.Dx() * img.Rect.Dy())

			checkExpectedAvg := func(t *testing.T, kernel Kernel) {
				t.Helper()

				result := kernel.Avg(img, 1, 1)

				if expected, actual := expectedAvg[0], int32(result.R); expected != actual {
					t.Errorf("Expected average of red channel to be %d but was %d", expected, actual)
				}
				if expected, actual := expectedAvg[1], int32(result.G); expected != actual {
					t.Errorf("Expected average of green channel to be %d but was %d", expected, actual)
				}
				if expected, actual := expectedAvg[2], int32(result.B); expected != actual {
					t.Errorf("Expected average of blue channel to be %d but was %d", expected, actual)
				}
				if expected, actual := expectedAvg[3], int32(result.A); expected != actual {
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
			totalWeight := int32(0)
			kernel := KernelWithRadius(1)
			for i := 0; i < kernel.SideLength(); i++ {
				for j := 0; j < kernel.SideLength(); j++ {
					weight := int32(i + j)
					totalWeight += weight
					kernel.SetWeightUniform(j, i, weight)
				}
			}

			avg := [4]int32{}
			for row, i := int32(0), img.Rect.Min.Y; i < img.Rect.Max.Y; row, i = row+1, i+1 {
				for col, j := int32(0), img.Rect.Min.X; j < img.Rect.Max.X; col, j = col+1, j+1 {
					c := img.NRGBAAt(j, i)
					avg[0] += int32(c.R) * (row + col)
					avg[1] += int32(c.G) * (row + col)
					avg[2] += int32(c.B) * (row + col)
					avg[3] += int32(c.A) * (row + col)
				}
			}
			avg[0] /= totalWeight
			avg[1] /= totalWeight
			avg[2] /= totalWeight
			avg[3] /= totalWeight

			result := kernel.Avg(img, 1, 1)

			if expected, actual := avg[0], int32(result.R); expected != actual {
				t.Errorf("Expected average of red channel to be %d but was %d", expected, actual)
			}
			if expected, actual := avg[1], int32(result.G); expected != actual {
				t.Errorf("Expected average of green channel to be %d but was %d", expected, actual)
			}
			if expected, actual := avg[2], int32(result.B); expected != actual {
				t.Errorf("Expected average of blue channel to be %d but was %d", expected, actual)
			}
			if expected, actual := avg[3], int32(result.A); expected != actual {
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

			checkExpectedMax := func(t *testing.T, kernel Kernel, uniformWeight int) {
				t.Helper()

				expectedMax := [4]int32{-1, -1, -1, -1}

				for i := img.Rect.Min.Y; i < img.Rect.Max.Y; i++ {
					for j := img.Rect.Min.X; j < img.Rect.Max.X; j++ {
						c := img.NRGBAAt(j, i)

						if uniformWeight > 0 {
							if int32(c.R) > expectedMax[0] || expectedMax[0] < 0 {
								expectedMax[0] = int32(c.R)
							}
							if int32(c.G) > expectedMax[1] || expectedMax[1] < 0 {
								expectedMax[1] = int32(c.G)
							}
							if int32(c.B) > expectedMax[2] || expectedMax[2] < 0 {
								expectedMax[2] = int32(c.B)
							}
							if int32(c.A) > expectedMax[3] || expectedMax[3] < 0 {
								expectedMax[3] = int32(c.A)
							}
						} else if uniformWeight < 0 {
							if int32(c.R) < expectedMax[0] || expectedMax[0] < 0 {
								expectedMax[0] = int32(c.R)
							}
							if int32(c.G) < expectedMax[1] || expectedMax[1] < 0 {
								expectedMax[1] = int32(c.G)
							}
							if int32(c.B) < expectedMax[2] || expectedMax[2] < 0 {
								expectedMax[2] = int32(c.B)
							}
							if int32(c.A) < expectedMax[3] || expectedMax[3] < 0 {
								expectedMax[3] = int32(c.A)
							}
						}
					}
				}

				result := kernel.Max(img, 1, 1)

				if expected, actual := expectedMax[0], int32(result.R); expected != actual {
					t.Errorf("Expected max of red channel to be %d but was %d", expected, actual)
				}
				if expected, actual := expectedMax[1], int32(result.G); expected != actual {
					t.Errorf("Expected max of green channel to be %d but was %d", expected, actual)
				}
				if expected, actual := expectedMax[2], int32(result.B); expected != actual {
					t.Errorf("Expected max of blue channel to be %d but was %d", expected, actual)
				}
				if expected, actual := expectedMax[3], int32(result.A); expected != actual {
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

				checkExpectedMax(t, kernel, 1)
			})

			t.Run("clips kernel against edges of image", func(t *testing.T) {
				kernel := KernelWithRadius(2)
				for i := 0; i < kernel.SideLength(); i++ {
					for j := 0; j < kernel.SideLength(); j++ {
						kernel.SetWeightUniform(j, i, -1)
					}
				}

				checkExpectedMax(t, kernel, -1)
			})
		})

		t.Run("ignores pixel values with zero weight", func(t *testing.T) {
			weights := []int32{
				0, 1, 0,
				1, 0, 1,
				0, 1, 0,
			}

			kernel := KernelWithRadius(1)
			kernel.SetWeightsUniform(weights)

			max := [4]int32{}

			for row, i := int32(0), img.Rect.Min.Y; i < img.Rect.Max.Y; row, i = row+1, i+1 {
				for col, j := int32(0), img.Rect.Min.X; j < img.Rect.Max.X; col, j = col+1, j+1 {
					w := weights[int(row)*kernel.SideLength()+int(col)]
					if w == 0 {
						continue
					}

					c := img.NRGBAAt(j, i)
					if int32(c.R) > max[0] {
						max[0] = int32(c.R)
					}
					if int32(c.G) > max[1] {
						max[1] = int32(c.G)
					}
					if int32(c.B) > max[2] {
						max[2] = int32(c.B)
					}
					if int32(c.A) > max[3] {
						max[3] = int32(c.A)
					}
				}
			}

			result := kernel.Max(img, 1, 1)

			if expected, actual := max[0], int32(result.R); expected != actual {
				t.Errorf("Expected max of red channel to be %d but was %d", expected, actual)
			}
			if expected, actual := max[1], int32(result.G); expected != actual {
				t.Errorf("Expected max of green channel to be %d but was %d", expected, actual)
			}
			if expected, actual := max[2], int32(result.B); expected != actual {
				t.Errorf("Expected max of blue channel to be %d but was %d", expected, actual)
			}
			if expected, actual := max[3], int32(result.A); expected != actual {
				t.Errorf("Expected max of alpha channel to be %d but was %d", expected, actual)
			}
		})
	})

	t.Run("Min()", func(t *testing.T) {
		img := randomImage(3, 3)

		t.Run("with uniform weights", func(t *testing.T) {

			checkExpectedMin := func(t *testing.T, kernel Kernel, uniformWeight int) {
				t.Helper()

				expectedMin := [4]int32{-1, -1, -1, -1}

				for i := img.Rect.Min.Y; i < img.Rect.Max.Y; i++ {
					for j := img.Rect.Min.X; j < img.Rect.Max.X; j++ {
						c := img.NRGBAAt(j, i)

						if uniformWeight > 0 {
							if int32(c.R) < expectedMin[0] || expectedMin[0] < 0 {
								expectedMin[0] = int32(c.R)
							}
							if int32(c.G) < expectedMin[1] || expectedMin[1] < 0 {
								expectedMin[1] = int32(c.G)
							}
							if int32(c.B) < expectedMin[2] || expectedMin[2] < 0 {
								expectedMin[2] = int32(c.B)
							}
							if int32(c.A) < expectedMin[3] || expectedMin[3] < 0 {
								expectedMin[3] = int32(c.A)
							}
						} else if uniformWeight < 0 {
							if -int32(c.R) < -expectedMin[0] || expectedMin[0] < 0 {
								expectedMin[0] = int32(c.R)
							}
							if -int32(c.G) < -expectedMin[1] || expectedMin[1] < 0 {
								expectedMin[1] = int32(c.G)
							}
							if -int32(c.B) < -expectedMin[2] || expectedMin[2] < 0 {
								expectedMin[2] = int32(c.B)
							}
							if -int32(c.A) < -expectedMin[3] || expectedMin[3] < 0 {
								expectedMin[3] = int32(c.A)
							}
						}
					}
				}

				result := kernel.Min(img, 1, 1)

				if expected, actual := expectedMin[0], int32(result.R); expected != actual {
					t.Errorf("Expected min of red channel to be %d but was %d", expected, actual)
				}
				if expected, actual := expectedMin[1], int32(result.G); expected != actual {
					t.Errorf("Expected min of green channel to be %d but was %d", expected, actual)
				}
				if expected, actual := expectedMin[2], int32(result.B); expected != actual {
					t.Errorf("Expected min of blue channel to be %d but was %d", expected, actual)
				}
				if expected, actual := expectedMin[3], int32(result.A); expected != actual {
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

				checkExpectedMin(t, kernel, 1)
			})

			t.Run("clips kernel against edges of image", func(t *testing.T) {
				kernel := KernelWithRadius(2)
				for i := 0; i < kernel.SideLength(); i++ {
					for j := 0; j < kernel.SideLength(); j++ {
						kernel.SetWeightUniform(j, i, -1)
					}
				}

				checkExpectedMin(t, kernel, -1)
			})
		})

		t.Run("ignores pixel values with zero weight", func(t *testing.T) {
			weights := []int32{
				0, 1, 0,
				1, 0, 1,
				0, 1, 0,
			}

			kernel := KernelWithRadius(1)
			kernel.SetWeightsUniform(weights)

			min := [4]int32{255, 255, 255, 255}

			for row, i := int32(0), img.Rect.Min.Y; i < img.Rect.Max.Y; row, i = row+1, i+1 {
				for col, j := int32(0), img.Rect.Min.X; j < img.Rect.Max.X; col, j = col+1, j+1 {
					w := weights[int(row)*kernel.SideLength()+int(col)]
					if w == 0 {
						continue
					}

					c := img.NRGBAAt(j, i)
					if int32(c.R) < min[0] {
						min[0] = int32(c.R)
					}
					if int32(c.G) < min[1] {
						min[1] = int32(c.G)
					}
					if int32(c.B) < min[2] {
						min[2] = int32(c.B)
					}
					if int32(c.A) < min[3] {
						min[3] = int32(c.A)
					}
				}
			}

			result := kernel.Min(img, 1, 1)

			if expected, actual := min[0], int32(result.R); expected != actual {
				t.Errorf("Expected min of red channel to be %d but was %d", expected, actual)
			}
			if expected, actual := min[1], int32(result.G); expected != actual {
				t.Errorf("Expected min of green channel to be %d but was %d", expected, actual)
			}
			if expected, actual := min[2], int32(result.B); expected != actual {
				t.Errorf("Expected min of blue channel to be %d but was %d", expected, actual)
			}
			if expected, actual := min[3], int32(result.A); expected != actual {
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
