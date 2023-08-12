import torch
import torchvision
import torch.onnx

# An instance of your model
# model = torchvision.models.resnet18()
model = torchvision.models.mobilenet_v3_large(pretrained=True)
model.classifier = model.classifier[:-1]
model.eval()

# An example input you would normally provide to your model's forward() method
x = torch.rand(1, 3, 224, 224)

# Export the model
torch_out = torch.onnx._export(model, x, "mobilenet_v3_large.onnx", export_params=True)
