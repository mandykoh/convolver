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
	for i := 0; i < kernel.SideLength(); i++ {
		for j := 0; j < kernel.SideLength(); j++ {
			offset := i*kernel.SideLength() + j
			kernel.SetWeightRGBA(j, i, weights[offset])
		}
	}

	for threadCount := 1; threadCount <= runtime.NumCPU(); threadCount++ {
		b.Run(fmt.Sprintf("with parallelism %d", threadCount), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = kernel.ApplySum(inputImg, threadCount)
			}
		})
	}
}

func TestApply(t *testing.T) {
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
}

func TestColourSeparation(t *testing.T) {
	imgFile, err := os.Open("test-images/avocado.png")
	if err != nil {
		t.Fatalf("Error opening input image: %v", err)
	}
	defer imgFile.Close()

	img, err := png.Decode(imgFile)
	if err != nil {
		t.Fatalf("Error decoding PNG: %v", err)
	}

	var inputImg *image.NRGBA
	inputImg, ok := img.(*image.NRGBA)
	if !ok {
		inputImg = image.NewNRGBA(img.Bounds())
		draw.Draw(inputImg, inputImg.Bounds(), img, image.Point{}, draw.Src)
	}

	kernel := KernelWithRadius(0)
	kernel.SetWeight(0, 0, 0, 0, 1, 1)

	startTime := time.Now()
	result := inputImg
	result = kernel.ApplySum(result, 4)
	endTime := time.Now()

	log.Printf("Finished in %v", endTime.Sub(startTime))

	outFile, err := os.Create("output.png")
	if err != nil {
		log.Fatalf("Error creating output file: %v", err)
	}
	defer outFile.Close()

	err = png.Encode(outFile, result)
	if err != nil {
		log.Fatalf("Error encoding output image: %v", err)
	}
}

func TestGaussianBlur(t *testing.T) {
	imgFile, err := os.Open("test-images/avocado.png")
	if err != nil {
		t.Fatalf("Error opening input image: %v", err)
	}
	defer imgFile.Close()

	img, err := png.Decode(imgFile)
	if err != nil {
		t.Fatalf("Error decoding PNG: %v", err)
	}

	var inputImg *image.NRGBA
	inputImg, ok := img.(*image.NRGBA)
	if !ok {
		inputImg = image.NewNRGBA(img.Bounds())
		draw.Draw(inputImg, inputImg.Bounds(), img, image.Point{}, draw.Src)
	}

	kernel := KernelWithRadius(2)

	weights := []int32{
		1, 4, 6, 4, 1,
		4, 16, 24, 16, 4,
		6, 24, 36, 24, 6,
		4, 16, 24, 16, 4,
		1, 4, 6, 4, 1,
	}
	for i := 0; i < kernel.SideLength(); i++ {
		for j := 0; j < kernel.SideLength(); j++ {
			offset := i*kernel.SideLength() + j
			kernel.SetWeightRGBA(j, i, weights[offset])
		}
	}

	startTime := time.Now()
	result := inputImg
	for i := 0; i < 8; i++ {
		result = kernel.ApplySum(result, 4)
	}
	endTime := time.Now()

	log.Printf("Finished in %v", endTime.Sub(startTime))

	outFile, err := os.Create("output.png")
	if err != nil {
		log.Fatalf("Error creating output file: %v", err)
	}
	defer outFile.Close()

	err = png.Encode(outFile, result)
	if err != nil {
		log.Fatalf("Error encoding output image: %v", err)
	}
}

func TestSharpen(t *testing.T) {
	imgFile, err := os.Open("test-images/avocado.png")
	if err != nil {
		t.Fatalf("Error opening input image: %v", err)
	}
	defer imgFile.Close()

	img, err := png.Decode(imgFile)
	if err != nil {
		t.Fatalf("Error decoding PNG: %v", err)
	}

	var inputImg *image.NRGBA
	inputImg, ok := img.(*image.NRGBA)
	if !ok {
		inputImg = image.NewNRGBA(img.Bounds())
		draw.Draw(inputImg, inputImg.Bounds(), img, image.Point{}, draw.Src)
	}

	kernel := KernelWithRadius(1)

	weights := []int32{
		0, -1, 0,
		-1, 5, -1,
		0, -1, 0,
	}
	for i := 0; i < kernel.SideLength(); i++ {
		for j := 0; j < kernel.SideLength(); j++ {
			offset := i*kernel.SideLength() + j
			kernel.SetWeightRGBA(j, i, weights[offset])
		}
	}

	startTime := time.Now()
	result := kernel.ApplySum(inputImg, 4)
	endTime := time.Now()

	log.Printf("Finished in %v", endTime.Sub(startTime))

	outFile, err := os.Create("output.png")
	if err != nil {
		log.Fatalf("Error creating output file: %v", err)
	}
	defer outFile.Close()

	err = png.Encode(outFile, result)
	if err != nil {
		log.Fatalf("Error encoding output image: %v", err)
	}
}

func TestDilateErode(t *testing.T) {
	imgFile, err := os.Open("test-images/convolver-alpha-1024.png")
	if err != nil {
		t.Fatalf("Error opening input image: %v", err)
	}
	defer imgFile.Close()

	img, err := png.Decode(imgFile)
	if err != nil {
		t.Fatalf("Error decoding PNG: %v", err)
	}

	var inputImg *image.NRGBA
	inputImg, ok := img.(*image.NRGBA)
	if !ok {
		inputImg = image.NewNRGBA(img.Bounds())
		draw.Draw(inputImg, inputImg.Bounds(), img, image.Point{}, draw.Src)
	}

	kernel := KernelWithRadius(2)

	weights := []int32{
		0, 1, 1, 1, 0,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		0, 1, 1, 1, 0,
	}
	for i := 0; i < kernel.SideLength(); i++ {
		for j := 0; j < kernel.SideLength(); j++ {
			offset := i*kernel.SideLength() + j
			kernel.SetWeightRGBA(j, i, weights[offset])
		}
	}

	startTime := time.Now()
	result := inputImg
	result = kernel.ApplyMax(result, 4)
	result = kernel.ApplyMin(result, 4)
	endTime := time.Now()

	log.Printf("Finished in %v", endTime.Sub(startTime))

	outFile, err := os.Create("output.png")
	if err != nil {
		log.Fatalf("Error creating output file: %v", err)
	}
	defer outFile.Close()

	err = png.Encode(outFile, result)
	if err != nil {
		log.Fatalf("Error encoding output image: %v", err)
	}
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
