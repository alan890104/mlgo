package main

import (
	"fmt"
	"mlgo/ml"
	"os"
)

func modelEval(fname string, threadCount int, inputData []float32) *ml.Tensor {
	file, err := os.Open(fname)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	defer file.Close()

	ctx0 := &ml.Context{}
	graph := ml.Graph{ThreadsCount: threadCount}

	module := ml.NewModuleImpl(file)
	fc1_Gemm := module.Gemm(ctx0, 784, 128)
	relu_Relu := module.Relu(ctx0)
	fc2_Gemm := module.Gemm(ctx0, 128, 10)

	Gemm_0 := ml.NewTensor1D(nil, ml.TYPE_F32, uint32(784))
	copy(Gemm_0.Data, inputData)

	// pass forward
	
	fc1_Gemm_output_0 := fc1_Gemm(Gemm_0)
	relu_Relu_output_0 := relu_Relu(fc1_Gemm_output_0)
	output := fc2_Gemm(relu_Relu_output_0)


	// Run the computation
	ml.BuildForwardExpand(&graph, output)
	ml.GraphCompute(ctx0, &graph)

	return output
}

func main() {
	modelWeightsFname := "./ggml/simple_model.bin"
	inputData := make([]float32, 784)
	outputTensor := modelEval(modelWeightsFname, 1, inputData)
	ml.PrintTensor(outputTensor, "final tensor")
}