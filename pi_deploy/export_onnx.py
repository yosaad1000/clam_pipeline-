"""
export_onnx.py — Re-export resnet50_trunc.onnx from the correct .pth weights.
Run in the clam conda env:
    conda run -n clam python pi_deploy/export_onnx.py
"""
import sys
import torch
import torch.nn as nn
import numpy as np

print("Loading timm...", flush=True)
import timm

WEIGHTS = "pi_deploy/checkpoints/resnet50_timm.pth"
OUT     = "pi_deploy/resnet50_trunc.onnx"

# Exact same architecture as PyTorchResNet50
backbone = timm.create_model(
    "resnet50",
    features_only=True,
    out_indices=(3,),
    pretrained=False,
    num_classes=0,
)
state = torch.load(WEIGHTS, map_location="cpu")
missing, unexpected = backbone.load_state_dict(state, strict=False)
print(f"Weights loaded — missing={len(missing)}  unexpected={len(unexpected)}", flush=True)
backbone.eval()

# Wrap GAP into the model so ONNX output is [N, 1024] not [N, 1024, 14, 14]
class ResNet50Trunc(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feat = self.backbone(x)[0]                  # [N, 1024, 14, 14]
        return self.pool(feat).squeeze(-1).squeeze(-1)  # [N, 1024]

wrapped = ResNet50Trunc(backbone)
wrapped.eval()

dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    pt_out = wrapped(dummy)
print(f"PyTorch output: shape={list(pt_out.shape)}  mean={pt_out.mean():.6f}  std={pt_out.std():.6f}", flush=True)

# Export
print(f"Exporting → {OUT}", flush=True)
torch.onnx.export(
    wrapped,
    dummy,
    OUT,
    input_names=["input"],
    output_names=["features"],
    dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
    opset_version=13,
    do_constant_folding=True,
)
print("Export done.", flush=True)

# Verify with onnxruntime
import onnxruntime as ort
sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
match   = np.allclose(pt_out.numpy(), onnx_out, atol=1e-4)
maxdiff = np.abs(pt_out.numpy() - onnx_out).max()
print(f"ONNX verification: match={match}  maxdiff={maxdiff:.8f}", flush=True)

if not match:
    print("ERROR: ONNX output does not match PyTorch — do not use this file.", flush=True)
    sys.exit(1)

print("SUCCESS — resnet50_trunc.onnx is consistent with resnet50_timm.pth", flush=True)
