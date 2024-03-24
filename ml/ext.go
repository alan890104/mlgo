package ml

import (
	"log"
	"os"
)

type Layer func(input *Tensor) *Tensor

type Module interface {
	Gemm(ctx *Context, inputSize uint32, outputSize uint32) Layer
	Relu(ctx *Context) Layer
	MaxPool1D(ctx *Context, kernelSize uint32, stride uint32, padding uint32) Layer
	Conv1D(ctx *Context, inputSize uint32, outputSize uint32, kernelSize uint32, stride uint32, padding uint32) Layer
	Conv2D(ctx *Context, inputSize uint32, outputSize uint32, kernelSize uint32, stride uint32, padding uint32) Layer
}

type ModuleImpl struct {
	fp *os.File
}

func NewModuleImpl(fp *os.File) Module {
	magic := readInt(fp)
	// 0x6F7261 is ora in hex
	if magic != 0x6F7261 {
		log.Fatal("invalid model file (bad magic)")
	}
	return &ModuleImpl{fp: fp}
}

func Passthrough(ctx *Context) Layer {
	return func(input *Tensor) *Tensor {
		return input
	}
}

func (f *ModuleImpl) Relu(ctx *Context) Layer {
	return func(input *Tensor) *Tensor {
		return Relu(ctx, input)
	}
}

// Conv1D implements ModelLoader.
func (f *ModuleImpl) Conv1D(ctx *Context, input_size uint32, kernel_size uint32, stride uint32, padding uint32, dilation uint32) Layer {
	return func(input *Tensor) *Tensor {
		return nil
	}
}

// Conv2D implements ModelLoader.
func (f *ModuleImpl) Conv2D(ctx *Context, input_size uint32, output_size uint32, kernel_size uint32, stride uint32, padding uint32) Layer {
	panic("unimplemented")
}

// MaxPool1D implements Module.
func (f *ModuleImpl) MaxPool1D(ctx *Context, kernelSize uint32, stride uint32, padding uint32) Layer {
	panic("unimplemented")
}

// Gemm implements ModelLoader.
func (f *ModuleImpl) Gemm(ctx *Context, input_size uint32, output_size uint32) Layer {
	w := NewTensor2D(nil, TYPE_F32, input_size, output_size)
	for i := 0; i < len(w.Data); i++ {
		w.Data[i] = readFP32(f.fp)
	}

	b := NewTensor1D(nil, TYPE_F32, output_size)
	for i := 0; i < len(b.Data); i++ {
		b.Data[i] = readFP32(f.fp)
	}

	return func(input *Tensor) *Tensor {
		return Add(ctx, MulMat(ctx, w, input), b)
	}
}
