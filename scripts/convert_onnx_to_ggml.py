import argparse
import struct
import onnx


def convert_onnx_to_ggml(onnx_file_path, output_ggml_path):
    IS_BIGENDIAN = True
    pack_fmt = "i" if not IS_BIGENDIAN else "!i"

    # create model weight file
    onnx_model_with_shape = onnx.shape_inference.infer_shapes(onnx.load(onnx_file_path))

    # create model weight file
    with open(output_ggml_path, "wb") as file:
        file.write(struct.pack("i", 0x6F7261))
        for initializer in onnx_model_with_shape.graph.initializer:
            weight = onnx.numpy_helper.to_array(initializer)
            weight.astype(">f4")
            weight.tofile(file)
    print(f"Done. Output file: {output_ggml_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to GGML format.")
    parser.add_argument(
        "onnx_file_path", type=str, help="Path to the input ONNX model file"
    )
    parser.add_argument(
        "output_ggml_path", type=str, help="Path for the output GGML file"
    )

    args = parser.parse_args()
    convert_onnx_to_ggml(args.onnx_file_path, args.output_ggml_path)


if __name__ == "__main__":
    main()
