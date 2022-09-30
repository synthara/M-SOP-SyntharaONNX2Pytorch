import onnx
from sympy import false
import torch

from onnx_pytorch.op_code_generators import OpCodeGenerator


class GemmOpCodeGenerator(OpCodeGenerator):

  def __init__(self,
               onnx_ver=onnx.defs.onnx_opset_version(),
               torch_ver=torch.__version__):
    super(GemmOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

  def gen(self, node, value_infos, initializers):
    attr_value_dict = self.get_attr_value_dict(node)
    inputs_str, outputs_str = self.gen_input_output_string(
        node, initializers, self.rename_helper, self.tensor_inplace)

    init_str, forward_str = [], []

    is_layer = len(node.input) == 3 and \
                node.input.__getitem__(0) not in initializers and \
                node.input.__getitem__(1) in initializers and \
                node.input.__getitem__(2) in initializers

    if is_layer:
      node_name = self.rename_helper.get_node_name(node.name, node.op_type) 
      weights = onnx.numpy_helper.to_array(initializers[node.input[1]])

      params_str = self.gen_params_str(
        out_features=weights.shape[0],
        in_features=weights.shape[1],
        bias= True)

      init_str.append(f"self.{node_name} = nn.Linear(**{{{params_str}}})")
      init_str.append(f"self.{node_name}.weight.data = {inputs_str[1]}")
      if len(node.input) > 2:
        init_str.append(f"self.{node_name}.bias.data = {inputs_str[2]}.float()")

      forward_str.append(f"{outputs_str[0]} = self.{node_name}({inputs_str[0]})")
    else:
      if attr_value_dict["transA"] == 1:
        inputs_str[0] = f"torch.transpose({inputs_str[0]}, 0, 1)"
      if attr_value_dict["transB"] == 1:
        inputs_str[1] = f"torch.transpose({inputs_str[1]}, 0, 1)"

      if attr_value_dict['alpha'] != 1.0 and attr_value_dict['beta'] != 1.0:
        forward_str.append(
            f"{outputs_str[0]} = {attr_value_dict['alpha']} * torch.matmul({', '.join(inputs_str[:2])}) + {attr_value_dict['beta']} * {inputs_str[2]}"
        )
      elif attr_value_dict['alpha'] != 1.0 and attr_value_dict['beta'] == 1.0:
        forward_str.append(
            f"{outputs_str[0]} = {attr_value_dict['alpha']} * torch.matmul({', '.join(inputs_str[:2])}) + {inputs_str[2]}"
        )
      elif attr_value_dict['alpha'] == 1.0 and  attr_value_dict['beta'] != 1.0:
        forward_str.append(
            f"{outputs_str[0]} = torch.matmul({', '.join(inputs_str[:2])}) + {attr_value_dict['beta']} * {inputs_str[2]}"
        )
      else:
        forward_str.append(
            f"{outputs_str[0]} = torch.matmul({', '.join(inputs_str[:2])}) + {inputs_str[2]}"
        )
    return {"init": init_str, "forward": forward_str}
