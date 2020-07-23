package convolver

import (
	"image"
	"image/draw"
	"image/png"
	"log"
	"os"
	"testing"
	"time"
)

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
