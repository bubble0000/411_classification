from easydict import EasyDict
import yaml
import argparse
import os

from train import model_train
from test import compare_multiple_models
# from deploy.onnxtrt import run_trt


def get_logpath(config_path):
    log_dir = os.path.join(os.path.dirname(config_path), 'logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_path = os.path.join(log_dir, 'log.txt')
    return log_path


parser = argparse.ArgumentParser(description='PyTorch class Training or Testing')
parser.add_argument('--config_path', default='./experiment/test/config.yaml')

def main():
    args = parser.parse_args()
    with open(args.config_path,'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)

    if config.mode == 'train':
        train_log_path = get_logpath(args.config_path)
        S = model_train(config,train_log_path)    ##train
        S.train()

    elif config.mode == 'evaluate':
        res = compare_multiple_models(config,args.config_path)
        print(res)

    elif config.mode == 'tensorrt':
        cls_res,times = run_trt(config,None)
        print("label: ",cls_res)
        print("time consume: ", times)

if __name__=="__main__":
    main()
