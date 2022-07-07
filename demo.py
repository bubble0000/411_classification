from easydict import EasyDict
import yaml
import argparse
import cv2
import torch
from datasets.data_aug import transform_test
from test import class_net_test


def preprocess(config,image_path):
    img = cv2.imread(image_path)
    test_transform = transform_test(config.datasets.transform.test.kwargs['size'])
    img = test_transform(img)
    img = torch.unsqueeze(img,0)
    img = img.cuda()
    return img


def run(config,model_path,image_path):
    model = class_net_test(config,model_path)
    img = preprocess(config,image_path)
    res = model.test_img(img)

    class_idx_dict_cover = model.dataset_train.img_datas.class_to_idx
    class_idx_dict = {k: v for v, k in class_idx_dict_cover.items()}

    return class_idx_dict[int(res[-1])]


parser = argparse.ArgumentParser(description='PyTorch class Testing')
parser.add_argument('--config_path', default='./experiment/test/config.yaml')
parser.add_argument('--model_path', default='/media/a/新加卷1/Download/Programs/411_classification-master/experiment/test/models/0.pth')
parser.add_argument('--image_path', default='/media/a/新加卷1/Download/Programs/411_classification-master/test.jpg')

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    predicted_label = run(config,args.model_path,args.image_path)
    print(predicted_label)


