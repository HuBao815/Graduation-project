'''
Product Grad_Cam Heatmap
Paper https://arxiv.org/abs/1610.02391 
Copyright (c) Xiangzi Dai, 2020
'''
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import sys

from model import *
from Resnet import *
from matplotlib import pyplot as plt

def get_last_conv(m):
    """
    Get the last conv layer in an Module.
    """
    convs = filter(lambda k: isinstance(k, torch.nn.Conv2d), m.modules())
    # print('convs:', convs)
    # print('list(convs)[-1]:', list(convs)[-1])
    return list(convs)[-1]

class Grad_Cam:
    def __init__(self, model,target_layer_names, use_cuda):
        self.model = model
        self.target = target_layer_names
        self.use_cuda = use_cuda
        self.grad_val = []
        self.feature = [] #feature dim is same as grad_val
        self.hook = []
        self.img = []
        self.inputs = None
        self._register_hook()
    def get_grad(self,module,input,output):
            self.grad_val.append(output[0].detach())
    def get_feature(self,module,input,output):
            self.feature.append(output.detach())
    def _register_hook(self):
        for i in self.target:
                self.hook.append(i.register_forward_hook(self.get_feature))
                self.hook.append(i.register_backward_hook(self.get_grad))

    def _normalize(self,cam, img, img_path):
        h,w,c = self.inputs.shape

        # h, w, c = img.shape
        cam = (cam-np.min(cam))/np.max(cam)
        cam = cv2.resize(cam, (w,h))


        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        # test = cv2.imread(img_path)

        cam = heatmap + np.float32(self.inputs)
        # cam = heatmap + np.float32(img) / 255
        # plt.imshow(cam)
        # plt.show()

        cam = cam / np.max(cam)
        return np.uint8(255*cam)

    def remove_hook(self):
        for i in self.hook:
            i.remove()

    def _preprocess_image(self,img):
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

    def __call__(self, img,idx=None):
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
            idx = np.argmax(outputs_com.detach().cpu().numpy()) #predict id
            # idx = np.argmax(output.cpu().numpy())  # predict id
        target = outputs_com[0][idx]
        print("index:", idx+1)
        target.backward()
        # predicted_com.backward()

        #computer 
        weights = []
        for i in self.grad_val[::-1]: #i dim: [1,512,7,7]
             weights.append(np.mean(i.squeeze().cpu().numpy(),axis=(1,2)))
        for index,j in enumerate(self.feature):# j dim:[1,512,7,7]
             cam = (j.squeeze().cpu().numpy()*weights[index][:,np.newaxis,np.newaxis]).sum(axis=0)
             cam = np.maximum(cam,0) # relu
             self.img.append(self._normalize(cam, img, img_path))
        return self.img


if __name__ == '__main__':
    # model_path = sys.argv[1]
    # img_path = sys.argv[2]
    model_path = r'E:\testdata\model\model.pth'
    # img_path = 'E:/ubuntu20210117/A1/A1-dataset/mushroom_group/280_Mycena_galericulata_072657/32.jpg'
    img_path = r'E:\testdata\dataset\tun_class\test\class001\24010.JPG'
    # img_path = 'F:/mushroom/mushrooms/22mushroom/Bai_Du_E_Gao_Jun_2019/bdegj_time1.jpg'
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


    print('loading model...')
    model = torch.load(model_path)
    model.eval()


    # print(model.state_dict)
    img = cv2.imread(img_path, 1)  # 加载彩色图片，这个是默认参数，可以直接写1

    #todo 处理再传入model
    img = cv2.resize(img, (550, 550))
    # cv2.imwrite('r_0.jpg', img)
    img = img[51:51+448, 51:51+448]
    # cv2.imwrite('r.jpg',img)


    m = get_last_conv(model)
    target_layer = [m]
    Grad_cams = Grad_Cam(model, target_layer, use_cuda)
    grad_cam_list = Grad_cams(img)
    #target_layer corresponding grad_cam_list
    cv2.imwrite('24004_2.JPG', grad_cam_list[0])
    i = cv2.imread('24004_2.JPG')
    img = cv2.imread(img_path)
    #todo 处理再传入model
    img = cv2.resize(img, (550, 550))
    # cv2.imwrite('r_0.jpg', img)
    img = img[51:51+448, 51:51+448]
    # cv2.imwrite('r.jpg',img)


    img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_AREA)
    j = cv2.imread(img)
    # hmerge = np.hstack((i, img))
    # cv2.imshow('out', hmerge)
    cv2.waitKey()


# python test_grad_cam.py D:/1-2-code/Grad_Cam-pytorch-resnet50\best_model.pth E:\ubuntu20210117\1-1\1-10_datasets\kagglecatsanddogs\PetImages\Cat\166.jpg