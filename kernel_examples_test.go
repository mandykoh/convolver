package convolver_test

import (
	"github.com/mandykoh/convolver"
	"image/png"
	"log"
	"os"
	"path"
	"runtime"
	"time"
)

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

	kernel := convolver.KernelWithRadius(0)
	kernel.SetWeightRGBA(0, 0, 0, 0, 1, 1)

	startTime := time.Now()
	result := kernel.ApplyAvg(img, runtime.NumCPU())
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

	weights := []float32{
		0, 1, 1, 1, 0,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		0, 1, 1, 1, 0,
	}

	kernel := convolver.KernelWithRadius(2)
	kernel.SetWeightsUniform(weights)

	startTime := time.Now()
	result := img
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

func ExampleKernel_edgeDetect() {
	imgFile, err := os.Open("test-images/avocado.png")
	if err != nil {
		log.Panicf("Error opening input image: %v", err)
	}
	defer imgFile.Close()

	img, err := png.Decode(imgFile)
	if err != nil {
		log.Panicf("Error decoding PNG: %v", err)
	}

	weights := []float32{
		-1, -1, -1,
		-1, 8, -1,
		-1, -1, -1,
	}

	kernel := convolver.KernelWithRadius(1)
	kernel.SetWeightsUniform(weights)

	startTime := time.Now()
	result := kernel.ApplyAvg(img, runtime.NumCPU())
	endTime := time.Now()

	log.Printf("Edge detection applied in %v", endTime.Sub(startTime))

	_ = os.Mkdir("example-output", os.ModePerm)
	outFilePath := path.Join("example-output", "example-edge-detect.png")
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

	weights := []float32{
		1, 4, 6, 4, 1,
		4, 16, 24, 16, 4,
		6, 24, 36, 24, 6,
		4, 16, 24, 16, 4,
		1, 4, 6, 4, 1,
	}

	kernel := convolver.KernelWithRadius(2)
	kernel.SetWeightsUniform(weights)

	startTime := time.Now()
	result := img
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

	weights := []float32{
		0, -1, 0,
		-1, 5, -1,
		0, -1, 0,
	}

	kernel := convolver.KernelWithRadius(1)
	kernel.SetWeightsUniform(weights)

	startTime := time.Now()
	result := kernel.ApplyAvg(img, runtime.NumCPU())
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
