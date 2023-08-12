import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity


def init_model(resume_from_checkpoint=None, backbone='resnet'):
    # load transforms
    normalize = transforms.Normalize(
        mean=[0.5], std=[0.5]
    )
    input_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])

    if backbone == 'resnet50':
        # load pretrained model and drop fc layer
        resnet50 = models.resnet50(pretrained=True)
        resnet50.fc = nn.Sequential()

        saved_dict = torch.load(resume_from_checkpoint, map_location=torch.device('cuda:0'))
        new_dict = []
        for name, param in saved_dict.items():
            new_dict.append((name[7:], param))
        new_dict = OrderedDict(new_dict)

        if resume_from_checkpoint:
            resnet50.load_state_dict(new_dict)
        resnet50.eval()
        return (resnet50, input_transform)
    elif backbone == 'resnet34':
        # load pretrained model and drop fc layer
        resnet34 = models.resnet34(pretrained=True)
        resnet34.fc = nn.Sequential()

        saved_dict = torch.load(resume_from_checkpoint, map_location=torch.device('cuda:0'))
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
            mobilenet.load_state_dict(torch.load(resume_from_checkpoint, map_location=torch.device('cuda:0')))
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


class FoodSeeker:
    def __init__(self, backbone='mobilenet', weights='', embeddings_dir='', library_dir='', today_menu_idx=None,
                 dist='l2'):
        self.model, self.transforms = init_model(backbone=backbone, resume_from_checkpoint=weights)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
        self.dist = dist
        self.model.to(self.device)
        if not os.path.exists(embeddings_dir):
            os.makedirs(embeddings_dir)
            self.generate_embedding_files(library_dir, embeddings_dir)
        self.embeddings, self.class_indexes = self.read_vectors_from_npz(embeddings_dir=embeddings_dir)

    def read_vectors_from_npz(self, embeddings_dir):
        """ read vectors from txt file
        """
        embeddings = np.load(os.path.join(embeddings_dir, 'embeddings.npy'))
        class_indexes = np.load(os.path.join(embeddings_dir, 'class_indexes.npy'))
        return embeddings, class_indexes

    def generate_embedding_files(self, library_dir, output_dir):
        embeddings, class_index = [], []

        image_cates = os.listdir(library_dir)
        for cate in tqdm(image_cates):
            image_files = [x for x in os.listdir(os.path.join(library_dir, cate)) if x.endswith('.png') or x.endswith('.jpg')]
            for _img_f in image_files:
                cls_name = cate
                op = self.vector_extractor_on_path(os.path.join(library_dir, cate, _img_f))
                embeddings.append(op)
                class_index.append(cls_name)

        np.save(os.path.join(output_dir, 'embeddings.npy'), np.array(embeddings))
        np.save(os.path.join(output_dir, 'class_indexes.npy'), np.array(class_index))
        return

    def vector_extractor_on_img_obj(self, img_obj):
        """ tiny extractor.
        """
        img_object = img_obj.convert("RGB")
        with torch.no_grad():
            img_t = self.transforms(img_object)
            if torch.cuda.is_available():
                img_t = img_t.to(self.device)
            batch_t = torch.unsqueeze(img_t, 0)
            out = self.model(batch_t)
        return out[0].detach().cpu().numpy()

    def vector_extractor_on_array(self, img_array):
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_obj = Image.fromarray(img_array)
        with torch.no_grad():
            img_t = self.transforms(img_obj)
            if torch.cuda.is_available():
                img_t = img_t.to(self.device)
            batch_t = torch.unsqueeze(img_t, 0)
            out = self.model(batch_t)
        return out[0].detach().cpu()

    def vector_extractor_on_path(self, img_path):
        img_object = Image.open(img_path)
        return self.vector_extractor_on_img_obj(img_object)

    def predict(self, img_obj):
        output = self.vector_extractor_on_array(img_obj)
        return self.class_indexes[np.linalg.norm(output - self.embeddings, axis=1).argmin()]

    def predict_v2(self, img_obj):
        output = self.vector_extractor_on_array(img_obj)
        mutual_sim = cosine_similarity([output.numpy()], self.embeddings)
        _max_idx, _max_v = mutual_sim.argmax(), mutual_sim.max()
        return self.class_indexes[_max_idx], _max_v

    def compare(self, src_img_array, compare_folder):
        res = []
        output = self.vector_extractor_on_array(src_img_array)
        for cf in tqdm(os.listdir(compare_folder)):
            if cf.endswith(('jpg', 'png')):
                img_path = os.path.join(compare_folder, cf)
                c_array = cv2.imread(img_path)
                c_op = self.vector_extractor_on_array(c_array)
                res.append([np.linalg.norm(output - c_op), cf])
        res.sort()
        return


def main():
    fs = FoodSeeker(embeddings_dir=r'C:\worksp\xxcy\Torch2NCNN\data\SeekerLibrary\eval_230806\mobilenetv3_large_20',
                    library_dir=r'C:\worksp\xxcy\data\cls_data\feature_extractor\data_230804\xiaotu_en',
                    backbone='mobilenet_arcface',
                    weights=r'C:\worksp\arcface-pytorch\checkpoints\230806\mobilenetv3_large_20.pth')
    image_path1 = r'C:\worksp\xxcy\DetRice\data\images\bus.jpg'
    img_array1 = cv2.imread(image_path1)
    print(fs.predict(img_array1))


if __name__ == '__main__':
    main()
