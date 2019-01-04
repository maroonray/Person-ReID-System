import numpy as np
import pickle
from functools import partial
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
if __name__ == '__main__':
    import models
import PeronReid.models as models


def preprocess_img(img):
    # Convert BGR to RGB
    img = img[..., ::-1]
    # imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_mean = [123.675, 116.28, 103.53]
    # imagenet_std = [0.229, 0.224, 0.225]
    imagenet_std = [58.395, 57.12, 57.375]
    img = np.divide(np.subtract(img, imagenet_mean), imagenet_std)
    img = cv2.resize(img, (64, 160))
    img = np.moveaxis(img, -1, 0)
    img = np.asarray(img)
    img = torch.from_numpy(img).float().to('cuda:0')
    return img


def compare(query_feature, baseline_embeddings):
    pass


def load_reid_model(model_path, model_arch='hacnn'):
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(2018)
    model = models.init_model(name=model_arch, num_classes=751, use_gpu=True)
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    checkpoint = torch.load(model_path, pickle_module=pickle)
    pretrain_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    model = nn.DataParallel(model).cuda()
    model.eval()
    return model


def query_imgs(img, model):
    input = []
    for p in img:
        try:
            p = preprocess_img(p)
        except cv2.error:
            print('what??????')
        input.append(p)
    input = torch.stack(input)
    feature = model(input).cpu().detach().numpy()
    return feature


if __name__ == '__main__':
    model_path = '/home/yul-j/Desktop/Demos/deep-person-reid/models/hacnn_market_xent/hacnn_market_xent.pth.tar'
    img = cv2.imread('/home/yul-j/Desktop/Demos/deep-person-reid/data/market1501/query/0913_c5s2_113902_00.jpg')
    model = load_reid_model(model_path)
    img = preprocess_img(img)
    feature = model(img)
    compare(feature, feature)
