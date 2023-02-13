import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import cv2

# 下面的模块是根据所指定的模型筛选出指定层的特征图输出，
# 如果未指定也就是extracted_layers是None则以字典的形式输出全部的特征图，
# 另外因为全连接层本身是一维的没必要输出因此进行了过滤。

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                x = x.view(x.size(0), -1)

            x = module(x)
            print(name)
            if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
                outputs[name] = x

        return outputs


def get_picture(pic_name, transform):
    img = skimage.io.imread(pic_name)
    img = skimage.transform.resize(img, (256, 256))
    img = np.asarray(img, dtype=np.float32)
    return transform(img)


def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def get_feature():
    pic_dir = 'PATH/imgs/AAA.jpg'  
    transform = transforms.ToTensor()
    img = get_picture(pic_dir, transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 插入维度
    img = img.unsqueeze(0)

    img = img.to(device)

    # 这里主要是一些参数，比如要提取的网络，网络的权重，要提取的层，指定的图像放大的大小，存储路径等等。
    net = models.detection.MaskRCNN.to(device) 
    net.load_state_dict(torch.load('PATH/weights/XXX.pth'))
    exact_list = None
    dst = './feautures/'
    therd_size = 640

    myexactor = FeatureExtractor(net, exact_list)
    outs = myexactor(img)
    # 这段主要是存储图片，为每个层创建一个文件夹将特征图以JET的colormap进行按顺序存储到该文件夹，
    # 并且如果特征图过小也会对特征图放大同时存储原始图和放大后的图。
    for k, v in outs.items():
        features = v[0]
        iter_range = features.shape[0]
        for i in range(iter_range):
            # plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='jet')
            if 'fc' in k:
                continue

            feature = features.data.cpu().numpy()
            feature_img = feature[i, :, :]
            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)

            dst_path = os.path.join(dst, k)

            make_dirs(dst_path)
            feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
            if feature_img.shape[0] < therd_size:
                tmp_file = os.path.join(dst_path, str(i) + '_' + str(therd_size) + '.jpg')
                tmp_img = feature_img.copy()
                tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(tmp_file, tmp_img)

            dst_file = os.path.join(dst_path, str(i) + '.jpg')
            cv2.imwrite(dst_file, feature_img)


if __name__ == '__main__':
    get_feature()
