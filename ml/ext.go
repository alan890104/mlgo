package ml

import (
	"log"
	"os"
)

type Layer func(input *Tensor) *Tensor

type ACT int

const (
	ACT_RELU = iota
	ACT_SIGMOID
	ACT_TANH
)

type ModelLoader interface {
	GeMM(ctx *Context, input_size uint32, output_size uint32) Layer
	Activation(ctx *Context, act ACT) Layer
	Conv1D(ctx *Context, input_size uint32, output_size uint32, kernel_size uint32, stride uint32, padding uint32) Layer
	Conv2D(ctx *Context, input_size uint32, output_size uint32, kernel_size uint32, stride uint32, padding uint32) Layer
}

type FileModelLoaderImpl struct {
	fp *os.File
}

func NewFileModelLoaderImpl(fp *os.File) ModelLoader {
	return &FileModelLoaderImpl{fp: fp}
}

// Activation implements ModelLoader.
func (f *FileModelLoaderImpl) Activation(ctx *Context, act ACT) Layer {
	switch act {
	case ACT_RELU:
		return func(input *Tensor) *Tensor {
			return Relu(ctx, input)
		}
	case ACT_SIGMOID:
	case ACT_TANH:
	}
	return func(input *Tensor) *Tensor {
		log.Fatalf("invalid activation function: %d", act)
		return nil
	}
}

// Conv1D implements ModelLoader.
func (f *FileModelLoaderImpl) Conv1D(ctx *Context, input_size uint32, output_size uint32, kernel_size uint32, stride uint32, padding uint32) Layer {
	panic("unimplemented")
}

// Conv2D implements ModelLoader.
func (f *FileModelLoaderImpl) Conv2D(ctx *Context, input_size uint32, output_size uint32, kernel_size uint32, stride uint32, padding uint32) Layer {
	panic("unimplemented")
}

// GeMM implements ModelLoader.
func (f *FileModelLoaderImpl) GeMM(ctx *Context, input_size uint32, output_size uint32) Layer {
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
