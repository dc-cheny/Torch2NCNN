import os
import sys

import pandas as pd

sys.path.append(r'C:\worksp\xxcy\Torch2NCNN\EdgeDeploy')

from FoodSeeker import FoodSeeker
import cv2
import numpy as np
from pypinyin import lazy_pinyin

if __name__ == '__main__':
    fs = FoodSeeker(embeddings_dir=r'C:\worksp\xxcy\Torch2NCNN\data\SeekerLibrary\eval_230806\mobilenetv3_large_20',
                    backbone='mobilenet_arcface',
                    weights=r'C:\worksp\arcface-pytorch\checkpoints\230806\mobilenetv3_large_10.pth')

    detect_dir = r'C:\worksp\xxcy\data\cls_data\feature_extractor\data_230804\filtered\Error'
    meal_dir = os.listdir(detect_dir)
    corr_num = 0
    all_num = 0

    res = []
    for md in meal_dir:
        imgs = os.listdir(os.path.join(detect_dir, md))
        for img in imgs:
            img_path = os.path.join(detect_dir, md, img)
            img_array1 = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            max_idx, max_v = fs.predict_v2(img_array1)
            if ''.join(lazy_pinyin(md)) == max_idx:
                corr_num += 1
            all_num += 1
            res.append([md, img, max_idx, max_v])

    df = pd.DataFrame(res, columns=['meal_name', 'img_name', 'match_name', 'dist'])
    df.to_excel('dist_distribution.xlsx')
    print(corr_num / all_num, corr_num, all_num)
