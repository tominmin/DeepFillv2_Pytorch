import onnx 
from onnxsim import simplify
import os
import torch

# ref. https://github.com/daquexian/onnx-simplifier
model = onnx.load(os.path.join(os.path.dirname(__file__), "../../artifact/deepfillv2_mod.onnx"))
model_simp, check = simplify(model)

assert check, "Simplified ONNX model could not be validated"

dummy_img = torch.randn((1,3,320,320)).cpu()
dummy_mask = torch.randn((1,1,320,320)).cpu()
onnx.save(model_simp, os.path.join(os.path.dirname(__file__), "../../artifact/deepfillv2_mod_simp.onnx"))