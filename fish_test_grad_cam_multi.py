'''
Product Grad_Cam Heatmap
Paper https://arxiv.org/abs/1610.02391 
Copyright (c) Xiangzi Dai, 2020
'''
import os

import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import sys

from model import *
from Resnet import *
from matplotlib import pyplot as plt

from matplotlib.font_manager import *
myfont = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
plt.rcParams['axes.unicode_minus']=False


def get_last_conv(m):
    """
    Get the last conv layer in an Module.
    """
    convs = filter(lambda k: isinstance(k, torch.nn.Conv2d), m.modules())
    # print('convs:', convs)
    # print('list(convs)[-1]:', list(convs)[-1])
    return list(convs)[-1]


class Grad_Cam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.target = target_layer_names
        self.use_cuda = use_cuda
        self.grad_val = []
        self.feature = []  # feature dim is same as grad_val
        self.hook = []
        self.img = []
        self.inputs = None
        self._register_hook()

    def get_grad(self, module, input, output):
        self.grad_val.append(output[0].detach())

    def get_feature(self, module, input, output):
        self.feature.append(output.detach())

    def _register_hook(self):
        for i in self.target:
            self.hook.append(i.register_forward_hook(self.get_feature))
            self.hook.append(i.register_backward_hook(self.get_grad))

    def _normalize(self, cam, img, img_path, pred_str):
        h, w, c = self.inputs.shape

        # h, w, c = img.shape
        cam = (cam - np.min(cam)) / np.max(cam)
        cam = cv2.resize(cam, (w, h))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        cam = heatmap + np.float32(self.inputs)
        # cam = heatmap + np.float32(img) / 255
        # plt.imshow(cam)
        # plt.show()

        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)



        # 在图像上添加预测标签
        text = 'predict:' + pred_str
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cam, text, (50, 50), font, 1.0, (255, 255, 255), 2)

        # 左CAM，右原图
        hmerge = np.hstack((cam, np.float32(self.inputs)*255))

        # return np.uint8(255 * cam)
        return hmerge

    def remove_hook(self):
        for i in self.hook:
            i.remove()

    def _preprocess_image(self, img):
        # means = [0.485, 0.456, 0.406]
        # stds = [0.229, 0.224, 0.225]
        means = [0.5, 0.5, 0.5]
        stds = [0.5, 0.5, 0.5]

        preprocessed_img = img.copy()[:, :, ::-1]
        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
        preprocessed_img = \
            np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
        preprocessed_img = torch.from_numpy(preprocessed_img)
        preprocessed_img.unsqueeze_(0)
        input = preprocessed_img.requires_grad_(True)
        return input

    def __call__(self, img, idx=None):
        """
        :param inputs: [w,h,c]
        :param idx: class id
        :return: grad_cam img list
        """
        self.model.zero_grad()
        # self.inputs = np.float32(cv2.resize(img, (224, 224))) / 255
        # self.inputs = np.float32(cv2.resize(img, (448, 448))) / 255
        self.inputs = np.float32(img) / 255
        inputs = self._preprocess_image(self.inputs)
        if self.use_cuda:
            inputs = inputs.cuda()
            self.model = self.model.cuda()
        # inputs = Varible(inputs)
        # output = self.model(inputs)
        output_1, output_2, output_3, output_concat = self.model(inputs)
        outputs_com = output_1 + output_2 + output_3 + output_concat

        # _, predicted = torch.max(output_concat.data, 1)
        # _, predicted_com = torch.max(outputs_com.data, 1)
        if idx is None:
            idx = np.argmax(outputs_com.detach().cpu().numpy())  # predict id
            # idx = np.argmax(output.cpu().numpy())  # predict id
        target = outputs_com[0][idx]
        # 显示预测标签
        class_names = {'bg': 0, 'ti-1': 1, 'ti-13': 4, 'ti-5': 2, 'ti-9': 3}
        pred_str = list(class_names.keys())[list(class_names.values()).index(int(idx))]

        print("index:", pred_str)
        target.backward()
        # predicted_com.backward()

        # computer
        weights = []
        for i in self.grad_val[::-1]:  # i dim: [1,512,7,7]
            weights.append(np.mean(i.squeeze().cpu().numpy(), axis=(1, 2)))
        for index, j in enumerate(self.feature):  # j dim:[1,512,7,7]
            cam = (j.squeeze().cpu().numpy() * weights[index][:, np.newaxis, np.newaxis]).sum(axis=0)
            cam = np.maximum(cam, 0)  # relu
            self.img.append(self._normalize(cam, img, img_path, pred_str))
        return self.img


def addimg(imgs, size, layout):
    """
    多图可视化
    """
    # imgs为需要展示的图片数组
    # size为展示时每张图片resize的大小
    # layout为展示图片的布局例如（3，3）代表3行3列）
    w = layout[0]
    h = layout[1]
    x = imgs[0].shape[2]
    if w * h - len(imgs) > 0:
        null_img = np.zeros((size[0], size[1], x), dtype='uint8')
        # 注意这里的dtype需要声明为'uint8'，否则和图片矩阵拼接时会导致图片的矩阵失真
        null_img = null_img * 255
    # null_img用来填充当图片数量不足时，布局上缺少的部分
    for i in range(len(imgs)):
        # 和同学交流的过程中发现如果出现有的图片通道不足的时候，会出现合并问题
        # 思考了一下，使用下面这段代码将灰度图片等通道数不足的图片补充成3个通道就ok
        if len(imgs[i].shape) < 3:
            imgs[i] = np.expand_dims(imgs[i], axis=2)
            imgs[i] = np.concatenate((imgs[i], imgs[i], imgs[i]), axis=-1)
        imgs[i] = cv2.resize(imgs[i], size)
    for j in range(h):
        for k in range(w):
            if j * w + k > len(imgs) - 1:
                f = k
                while f < w:
                    if f == 0:
                        imgw = null_img
                    else:
                        imgw = np.hstack((imgw, null_img))
                    f = f + 1
                break
            if k == 0:
                imgw = imgs[j * w]
            else:
                imgw = np.hstack((imgw, imgs[j * w + k]))
            print(j * w + k)
        if j == 0:
            imgh = imgw
        else:
            imgh = np.vstack((imgh, imgw))
    return imgh


if __name__ == '__main__':
    # model_path = sys.argv[1]
    # img_path = sys.argv[2]
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = '/media/hz/A2/visual_result'

    model_path = '/media/hz/A12/1-1/fish_5_model.pth'
    # img_path = 'E:/ubuntu20210117/A1/A1-dataset/mushroom_group/280_Mycena_galericulata_072657/32.jpg'
    # img_path = 'E:/ubuntu20210117/1-1/1-10_datasets/mushroom599/mushroom_split/train/class_125'
    # img_path = 'E:/ubuntu20210117/1-1/1-10_datasets/mushroom599/test/10'
    img_path = '/media/hz/A2/A2-datasets/_split/valid/鳀-1'
    output_dir = os.path.join(BASE_DIR, '鳀-1')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    use_cuda = torch.cuda.is_available()
    # load model
    # checkpoint = torch.load(model_path)
    # model = models.resnet50(pretrained=False, num_classes=2)
    # model.load_state_dict(checkpoint['state_dict'])
    # model.eval()

    # checkpoint = torch.load(model_path)
    # model = resnet50()
    # model = PMG(model, 512, 598)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # print('loading model...')
    # model.eval()

    print('loading model...')
    model = torch.load(model_path)
    model.eval()


    count = 1
    out_img = []
    for pathi in os.listdir(img_path):
        pathi = os.path.join(img_path, pathi)
        img = cv2.imread(pathi, 1)  # 加载彩色图片，这个是默认参数，可以直接写1

        # todo 处理再传入model
        img = cv2.resize(img, (550, 550))
        # cv2.imwrite('r_0.jpg', img)
        img = img[51:51 + 448, 51:51 + 448]
        # cv2.imwrite('r.jpg',img)


        m = get_last_conv(model)
        target_layer = [m]
        Grad_cams = Grad_Cam(model, target_layer, use_cuda)
        grad_cam_list = Grad_cams(img)
        img_name = os.path.join(output_dir, 'out' + str(count) + '.jpg')
        out_img.append(str(img_name))
        count += 1
        # target_layer corresponding grad_cam_list
        cv2.imwrite(img_name, grad_cam_list[0])

    # CV2可视化
    files = os.listdir(output_dir)
    imgs = []
    for file in files:
        imgs.append(cv2.imread(os.path.join(output_dir + '/' + str(file))))
    imgout = addimg(imgs, (448, 224), (5, 20))
    # 拼接为大图后另存为
    # cv2.imshow('out', imgout)
    cv2.waitKey()
    output_img = os.path.join(output_dir + '/' + 'output_img.jpg')
    cv2.imwrite(output_img, imgout)

# python test_grad_cam.py D:/1-2-code/Grad_Cam-pytorch-resnet50\best_model.pth E:\ubuntu20210117\1-1\1-10_datasets\kagglecatsanddogs\PetImages\Cat\166.jpg
