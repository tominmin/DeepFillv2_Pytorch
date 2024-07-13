from onnxruntime.quantization import quantize_dynamic, QuantType
import os

src_model_path = os.path.join(os.path.dirname(__file__), "../../artifact/deepfillv2_mod_simp.onnx")
dst_model_path = os.path.join(os.path.dirname(__file__), "../../artifact/deepfillv2_mod_simp_int8.onnx")

quantized_model = quantize_dynamic(src_model_path, dst_model_path, weight_type=QuantType.QUInt8)