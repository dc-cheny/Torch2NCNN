import torch
from torch import nn
from torchvision import models, transforms
from collections import OrderedDict
import os

def init_model(resume_from_checkpoint=None, backbone='resnet'):
    # load transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    input_transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    if backbone == 'resnet50':
        # load pretrained model and drop fc layer
        resnet50 = models.resnet50(pretrained=True)
        resnet50.fc = nn.Sequential()
        if resume_from_checkpoint:
            resnet50.load_state_dict(torch.load(resume_from_checkpoint))
        resnet50.eval()
        return (resnet50, input_transform)
    elif backbone == 'resnet34':
        # load pretrained model and drop fc layer
        resnet34 = models.resnet34(pretrained=True)
        resnet34.fc = nn.Sequential()

        saved_dict = torch.load(resume_from_checkpoint, map_location=torch.device('cpu'))
        new_dict = []
        for name, param in saved_dict.items():
            new_dict.append((name[7:], param))
        new_dict = OrderedDict(new_dict)

        if resume_from_checkpoint:
            resnet34.load_state_dict(new_dict)
        resnet34.eval()

        return (resnet34, input_transform)
    elif backbone == 'shufflenet':
        # load pretrained model and drop fc layer
        shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
        shufflenet.fc = nn.Sequential()
        if resume_from_checkpoint:
            shufflenet.load_state_dict(torch.load(resume_from_checkpoint))
        shufflenet.eval()

        return (shufflenet, input_transform)
    elif backbone == 'mobilenet':
        mobilenet = models.mobilenet_v3_large(pretrained=True)
        mobilenet.classifier[-1] = nn.Linear(in_features=1280, out_features=205, bias=True)

        if resume_from_checkpoint:
            mobilenet.load_state_dict(torch.load(resume_from_checkpoint, map_location=torch.device('cpu')))
        mobilenet.eval()

        return (mobilenet, input_transform)
    elif backbone == 'mobilenet_arcface':
        mobilenet = models.mobilenet_v3_large(pretrained=True)
        mobilenet.classifier[-1] = nn.Sequential()

        saved_dict = torch.load(resume_from_checkpoint, map_location=torch.device('cpu'))
        new_dict = []
        for name, param in saved_dict.items():
            new_dict.append((name[7:], param))
        new_dict = OrderedDict(new_dict)

        if resume_from_checkpoint:
            mobilenet.load_state_dict(new_dict)
        mobilenet.eval()

        return (mobilenet, input_transform)

    else:
        return None


class FoodFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        mobilenet = models.mobilenet_v3_large(pretrained=True)
        mobilenet.classifier = mobilenet.classifier[:-1]

        mobilenet.eval()

        self.model = mobilenet

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    pth_path = r'C:\worksp\arcface-pytorch\checkpoints\230806\mobilenetv3_large_20.pth'
    onnx_path = r'C:\worksp\xxcy\Torch2NCNN\data\cls_models\mobilenet_arcface_epoch20_230807\mobilenet_arcface_epoch20_230807.onnx'
    m = init_model(pth_path, backbone='mobilenet_arcface')

    # make dir of the onnx
    if not os.path.exists(os.path.dirname(onnx_path)):
        os.mkdir(os.path.dirname(onnx_path))
    torch_out = torch.onnx.export(m[0], torch.rand(1, 3, 224, 224), onnx_path, export_params=True)
