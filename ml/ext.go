package ml

import (
	"os"
)

type Layer func(input *Tensor) *Tensor

type ACT int

const (
	ACT_RELU = iota
	ACT_SIGMOID
	ACT_TANH
)

type Module interface {
	GeMM(ctx *Context, input_size uint32, output_size uint32) Layer
	Relu(ctx *Context) Layer
	Conv1D(ctx *Context, input_size uint32, output_size uint32, kernel_size uint32, stride uint32, padding uint32) Layer
	Conv2D(ctx *Context, input_size uint32, output_size uint32, kernel_size uint32, stride uint32, padding uint32) Layer
}

type ModuleImpl struct {
	fp *os.File
}

func NewModuleImpl(fp *os.File) Module {
	return &ModuleImpl{fp: fp}
}

func (f *ModuleImpl) Relu(ctx *Context) Layer {
	return func(input *Tensor) *Tensor {
		return Relu(ctx, input)
	}
}

// Conv1D implements ModelLoader.
func (f *ModuleImpl) Conv1D(ctx *Context, input_size uint32, output_size uint32, kernel_size uint32, stride uint32, padding uint32) Layer {
	panic("unimplemented")
}

// Conv2D implements ModelLoader.
func (f *ModuleImpl) Conv2D(ctx *Context, input_size uint32, output_size uint32, kernel_size uint32, stride uint32, padding uint32) Layer {
	panic("unimplemented")
}

// GeMM implements ModelLoader.
func (f *ModuleImpl) GeMM(ctx *Context, input_size uint32, output_size uint32) Layer {
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
