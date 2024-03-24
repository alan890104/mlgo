# MLGO

MLGO is tensor library for machine learning in pure Golang that can run on MIPS.

The machine learning part of this project refers to the legendary [ggml.cpp](https://github.com/ggerganov/ggml) framework.


## Build

`pip install -r requirements.txt`

## Linear Model
See `examples/linear_model/`

1. Export the AI model into ONNX
2. Use `./scripts/convert_onnx_to_ggml.py` to convert ONNX to ggml
   
   ```bash
   python3 ./scripts/convert_onnx_to_ggml.py ./examples/linear_model/onnx/simple_model.onnx ./examples/linear_model/ggml/simple_model.bin
   ```
4. Run `code_generator.py` to generate `/dist/simple_model.go`

   ```bash
   mkdir dist
   python3 ./scripts/code_generator.py
   ```
6. Run following command to check this model work

   ```bash
   go run ./dist/simple_model.go
   ```
