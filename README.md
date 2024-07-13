## DeepFillv2_Pytorch (for convert onnx model)
forked from https://github.com/csqiangwen/DeepFillv2_Pytorch
Convert the deepfillv2 PyTorch model to ONNX and quantize it

### how to use
0. (init project)
```
$ rye sync
```

1. export model onnx
```
$ rye run python src/cli/export_model_onnx.py
```

2. simplifier use onnxsim
```
$ rye run python src/cli/onnx_simplifier.py
```

3. quantization (uint8)
```
$ rye run python src/cli/quantization_onnx.py
```