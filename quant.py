import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = './models/baseline_nano_.onnx'
model_quant = 'model_quant.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QInt8, nodes_to_exclude=['/conv1/Conv'])

