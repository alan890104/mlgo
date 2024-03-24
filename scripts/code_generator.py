# This code will generate go-lang program by onnx model.
from jinja2 import Environment, FileSystemLoader

template_dir = "."

env = Environment(loader=FileSystemLoader(template_dir))

template = env.get_template("model_temp.py.j2")

from model_graph import Graph
import onnx


def generate_eval_func(file, model_graph: Graph):

    graph = model_graph
    nodes = graph.nodes
    n_input = len(graph.input)
    input = graph.input

    context = {
        "nodes": nodes,
        "input_name": input[0].name.split("::")[-1],
        "n_input": input[0].type.tensor_type.shape.dim[0].dim_value,
        "inputs": [
            input[0].type.tensor_type.shape.dim[i].dim_value
            for i in range(1, n_input + 1)
        ],
        "model_weights_fname": "./examples/linear_model/ggml/simple_model.bin",
    }

    rendered_code = template.render(context)
    with open(file, "w") as f:
        f.write(rendered_code)
    f.close()


def main():
    # load onnx model
    model_name = "simple_model"
    model_onnx = onnx.load(f"./examples/linear_model/onnx/{model_name}.onnx")
    model_onnx_with_shape = onnx.shape_inference.infer_shapes(model_onnx)

    graph = Graph(model_onnx_with_shape.graph)
    fout = f"./dist/{model_name}.go"
    generate_eval_func(fout, graph)


if __name__ == "__main__":
    main()
