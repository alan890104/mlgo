import onnx
from onnx import numpy_helper

from pprint import pprint


class Node:
    def __init__(self, node, code_str, last_node=False):
        self.name = self.replace_string(node.name)
        self.op_type = node.op_type
        self.input_name = self.replace_string(node.input[0])
        self.output_name = "output" if last_node else self.replace_string(node.output[0])
        self.attribute = node.attribute
        self.code_str = self.replace_string(code_str)

    def replace_string(self, string):
        string = string.split("::")[-1]
        return string.replace("/", "_").lstrip("_")


# This is a Graph class that can store the onnx model graph information.
class Graph:
    def __init__(self, graph):
        self.input = graph.input
        self.output = graph.output
        self.initializer = graph.initializer
        self.value_infos = graph.value_info

        # Mapping of operation types to their corresponding information retrieval methods
        self.operation_map = {
            "Conv": self.get_conv_information,
            "Relu": self.get_relu_information,
            "Softmax": self.get_softmax_information,
            "MaxPool": self.get_maxpool_information,
            "Reshape": self.get_reshape_information,
            "Gemm": self.get_gemm_information,
            "Concat": self.get_concat_information,
            "Constant": self.get_constant_information,
            "Transpose": self.get_transpose_information,
        }
        self.nodes = self.iterate_nodes(graph.node)

    def iterate_nodes(self, nodes):
        node_list = []
        n_nodes = len(nodes)
        for i, node in enumerate(nodes):
            # Use the operation_map to call the appropriate method
            if node.op_type in self.operation_map:
                info_method = self.operation_map[node.op_type]
                node_info = info_method(node)
                node_list.append(
                    Node(
                        node,
                        node_info.get("code_str", None),
                        last_node=True if i == n_nodes - 1 else False,
                    )
                )
            else:
                print("Information method for this node type is not implemented.")
        return node_list

    def get_conv_information(self, node):
        # conv has 3 inputs: X, W, B
        # conv has 1 output: Y
        # conv has 6 attributes: dilations, group, kernel_shape, pads, strides, auto_pad
        conv_input = {}
        conv_output = {}
        conv_attributes = {}

        conv_input[node.input[0]] = self.get_dims_from_input(node.input[0])
        for input in node.input[1:]:
            conv_input[input] = self.get_dims_from_initializer(input)
        conv_output[node.output[0]] = self.get_dims_from_output(node.output[0])
        for attr in node.attribute:
            conv_attributes[attr.name] = attr.ints
        return {
            "input": conv_input,
            "output": conv_output,
            "attributes": conv_attributes,
        }

    def get_relu_information(self, node):
        # relu has 1 input: X
        # relu has 1 output: Y
        relu_input = {}
        relu_output = {}
        relu_input[node.input[0]] = self.get_dims_from_input(node.input[0])
        relu_output[node.output[0]] = self.get_dims_from_output(node.output[0])
        code_str = f"{node.name} := module.Relu(ctx0)"
        return {"input": relu_input, "output": relu_output, "code_str": code_str}

    def get_softmax_information(self, node):
        # softmax has 1 input: X
        # softmax has 1 output: Y
        # softmax has 1 attribute: axis
        softmax_input = {}
        softmax_output = {}
        softmax_attributes = {}
        softmax_input[node.input[0]] = self.get_dims_from_input(node.input[0])
        softmax_output[node.output[0]] = self.get_dims_from_output(node.output[0])
        for attr in node.attribute:
            softmax_attributes[attr.name] = attr.ints
        return {
            "input": softmax_input,
            "output": softmax_output,
            "attributes": softmax_attributes,
        }

    def get_maxpool_information(self, node):
        # maxpool has 1 input: X
        # maxpool has 2 output: Y, Indices
        # maxpool has 6 attributes: dilations, kernel_shape, pads, strides, auto_pad
        maxpool_input = {}
        maxpool_output = {}
        maxpool_attributes = {}
        maxpool_input[node.input[0]] = self.get_dims_from_input(node.input[0])
        for output in node.output:
            maxpool_output[output] = self.get_dims_from_output(output)
        for attr in node.attribute:
            maxpool_attributes[attr.name] = attr.ints
        return {
            "input": maxpool_input,
            "output": maxpool_output,
            "attributes": maxpool_attributes,
        }

    def get_reshape_information(self, node):
        # reshape has 2 input: X
        # reshape has 1 output: Y
        # reshape has 1 attribute: shape
        reshape_input = {}
        reshape_output = {}
        reshape_attributes = {}
        for i in range(2):
            reshape_input[node.input[i]] = self.get_dims_from_input(node.input[i])
        reshape_output[node.output[0]] = self.get_dims_from_output(node.output[0])
        for attr in node.attribute:
            reshape_attributes[attr.name] = attr.ints
        return {
            "input": reshape_input,
            "output": reshape_output,
            "attributes": reshape_attributes,
        }

    def get_gemm_information(self, node):
        # gemm has 3 input: A, B, C
        # gemm has 1 output: Y
        # gemm has 4 attributes
        gemm_input = {}
        gemm_output = {}
        gemm_attributes = {}
        gemm_input[node.input[0]] = self.get_dims_from_input(node.input[0])
        for input in node.input[1:]:
            gemm_input[input] = self.get_dims_from_initializer(input)
        gemm_output[node.output[0]] = self.get_dims_from_output(node.output[0])
        for attr in node.attribute:
            gemm_attributes[attr.name] = attr.ints

        code_str = f"{node.name} := module.Gemm(ctx0, {gemm_input[node.input[0]][1]}, {gemm_output[node.output[0]][1]})"
        return {
            "input": gemm_input,
            "output": gemm_output,
            "attributes": gemm_attributes,
            "code_str": code_str,
        }

    def get_concat_information(self, node):
        # concat has lot of inputs
        # concat has 1 output
        # concat has 1 attribute: axis
        concat_input = {}
        concat_output = {}
        concat_attributes = {}
        for i in range(len(node.input)):
            concat_input[node.input[i]] = self.get_dims_from_input(node.input[i])
        concat_output[node.output[0]] = self.get_dims_from_output(node.output[0])
        concat_attributes[node.attribute[0].name] = node.attribute[0].ints
        return {
            "input": concat_input,
            "output": concat_output,
            "attributes": concat_attributes,
        }

    def get_constant_information(self, node):
        # constant has 0 input
        # constant has 1 output
        # constant has 8 attribute
        constant_output = {}
        constant_attributes = {}
        constant_output[node.output[0]] = self.get_dims_from_output(node.output[0])
        for attr in node.attribute:
            if attr.name == "value":
                constant_attributes[attr.name] = {
                    "dims": attr.t.dims,
                    "raw_data": numpy_helper.to_array(attr.t),
                }
            else:
                constant_attributes[attr.name] = attr.ints
        return {"output": constant_output, "attributes": constant_attributes}

    def get_transpose_information(self, node):
        # transpose has 1 input: X
        # transpose has 1 output: Y
        # transpose has 1 attribute: perm
        transpose_input = {}
        transpose_output = {}
        transpose_attributes = {}
        transpose_input[node.input[0]] = self.get_dims_from_input(node.input[0])
        transpose_output[node.output[0]] = self.get_dims_from_output(node.output[0])
        transpose_attributes[node.attribute[0].name] = node.attribute[0].ints
        return {
            "input": transpose_input,
            "output": transpose_output,
            "attributes": transpose_attributes,
        }

    def get_dims_from_input(self, input_name):
        for input in self.input:
            if input.name == input_name:
                dims = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
                return dims
        for value_info in self.value_infos:
            if value_info.name == input_name:
                dims = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                return dims

    def get_dims_from_initializer(self, initializer_name):
        for initializer in self.initializer:
            if initializer.name == initializer_name:
                dims = initializer.dims
                raw_data = numpy_helper.to_array(initializer)
                # return {"dims": dims, "raw_data": raw_data}
                return dims

        for value_info in self.value_infos:
            if value_info.name == initializer_name:
                dims = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                return dims

    def get_dims_from_output(self, output_name):
        for output in self.output:
            if output.name == output_name:
                dims = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
                return dims
        for value_info in self.value_infos:
            if value_info.name == output_name:
                dims = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                return dims


def main():
    # load onnx model
    model_name = "simple_model"
    model_onnx = onnx.load(f"./onnx/{model_name}.onnx")
    model_onnx_with_shape = onnx.shape_inference.infer_shapes(model_onnx)

    # store graph information in to file
    with open(f"./graph/{model_name}.txt", "w") as f:
        f.write(str(model_onnx_with_shape.graph))
    graph = Graph(model_onnx_with_shape.graph)
    pprint(graph.iterate_nodes())


if __name__ == "__main__":
    main()
