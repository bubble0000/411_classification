from __future__ import print_function

import yaml
from easydict import EasyDict
import click
import cv2
import numpy as np
import torch
from grad_cam import GradCAM,GuidedBackPropagation

from model import build_model
from datasets.data_aug import transform_test


class visualize_grad_cam():
    def __init__(self,config_path,cuda):
        self.get_device(cuda)
        self.load_config(config_path)
        self.load_model()
        self.load_image()
        self.gcam = GradCAM(self.model)
        self.model_name = self.config['model']['arch']

    def get_device(self,cuda):
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        if cuda:
            current_device = torch.cuda.current_device()
            print("Running on the GPU:", torch.cuda.get_device_name(current_device))
        else:
            print("Running on the CPU")

    def load_config(self,config_path):
        with open(config_path, 'rb') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.config = EasyDict(self.config)
        self.image_size = self.config.datasets.transform.test.kwargs['size']
        review_config = self.config['review']
        self.model_path = review_config['model_path']
        self.image_path = review_config['image_path']
        self.target_layers = review_config['visualize_layers']

    def load_model(self):
        self.model = build_model(self.config['model'])
        model_dict = torch.load(self.model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(model_dict['net'])
        self.model.to(self.device)
        self.model.eval()

    def load_image(self):
        self.raw_image = cv2.imread(self.image_path)
        test_transform = transform_test(self.config.datasets.transform.test.kwargs['size'])
        img = test_transform(self.raw_image)
        image = torch.unsqueeze(img, 0)
        self.image = image.to(self.device)

    def run_gcam(self):
        predictions = self.gcam.forward(self.image)
        top_idx = predictions[0][1]
        for target_layer in self.target_layers:
            self.gcam.backward(idx=top_idx)
            region = self.gcam.generate(target_layer=target_layer)
            img_name = "{}-gradcam-{}-{}.png".format(self.model_name, target_layer,top_idx)
            self.save_gradcam(img_name,region)

    def save_gradcam(self,filename,region):
        h, w, _ = self.raw_image.shape
        gcam = cv2.resize(region, (w, h))
        gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
        gcam = gcam.astype(np.float) + self.raw_image.astype(np.float)
        gcam = gcam / gcam.max() * 255.0
        cv2.imwrite(filename, np.uint8(gcam))


@click.command()
@click.option("-c", "--config-path", type=str, required=True,default="../experiment/test/config.yaml")
@click.option("--cuda/--no-cuda", default=True)
def main(config_path,cuda):
    gcam_func = visualize_grad_cam(config_path,cuda)
    gcam_func.run_gcam()

if __name__ == "__main__":
    main()
