# MLGO

MLGO is tensor library for machine learning in pure Golang that can run on MIPS.

The machine learning part of this project refers to the legendary [ggml.cpp](https://github.com/ggerganov/ggml) framework.


## Build

`pip install -r requirements.txt`

## Linear Model
See `examples/linear_model/`

1. Export the AI model into ONNX
2. Use `./scripts/convert_onnx_to_ggml.py` to convert ONNX to ggml
3. Run `code_generator.py` to generate `/dist/simple_model.go` 
4. Run following command to check this model work


```bash
cd dist
go run simple_model.go

```
