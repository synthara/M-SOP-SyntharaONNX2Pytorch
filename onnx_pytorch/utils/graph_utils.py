def find_target_layer_of_quant_weight(pretty_onnx_graph, quant_name):
    """
    For every QuantizeLinear of weight data, there are 4 steps in the opgraph until the node of the correspoding layer.

    QuantizeLinear -> x -> DequantizeLinear -> layer
    """
    n = quant_name
    for _ in range(4):
        n = list(pretty_onnx_graph.successors(n))[0]
    return n
kk