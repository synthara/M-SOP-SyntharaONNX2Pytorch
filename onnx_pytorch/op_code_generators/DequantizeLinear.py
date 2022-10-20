import onnx
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class DequantizeLinearOpCodeGenerator(OpCodeGenerator):

    def __init__(self,
                 onnx_ver=onnx.defs.onnx_opset_version(),
                 torch_ver=torch.__version__):
        super(DequantizeLinearOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

    def gen(self, node, value_infos, initializers):
        attr_value_dict = self.get_attr_value_dict(node)
        inputs_str, outputs_str = self.gen_input_output_string(
            node, initializers, self.rename_helper, self.tensor_inplace)

        node_name = self.rename_helper.get_node_name(node.name, node.op_type)
        init_str, forward_str = [], []

        params_str = self.gen_params_str(
            axis=attr_value_dict["axis"])

        scale = inputs_str[1]
        zero_point = inputs_str[2]

        if any("quant" in s or "weight" in s for s in [inputs_str[0], outputs_str[0]]):
            init_str.append(f"self.{outputs_str[0]} = {inputs_str[0]}")
            initializers.update(
                {
                    outputs_str[0]: initializers[node.input[0]]
                }
            )
        elif node.output[0] in self.model_outputs:  # dequantize linear should only have one child
            init_str.extend((f"self.{node_name} = torch.quantization.QuantStub()",
                             f"self.{node_name}.scale = {scale}",
                             f"self.{node_name}.zero_point = {zero_point}"))
            forward_str.append(f"{outputs_str[0]} = self.{node_name}({inputs_str[0]})")

        else:
            forward_str.append(f"{outputs_str[0]} = {inputs_str[0]}")

        return {"init": init_str, "forward": forward_str}
