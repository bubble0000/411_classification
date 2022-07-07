## Introduction

------

#####          **411_classfication**是一个统一的分类网络框架，可用于各种分类网络的训练、测试，并且可完成`pytorch->onnx->tensorrt`路线的模型转换。

## Requirements

------

#####           **python 3.6+**

#####           **torch 1.5.0+**

#####           onnx 1.4.0

#####           tensorrt 5.1.5.0

## Usage

------

###        **Train** 

#####                   `sh experiment/test/train.sh`



###        Evaluate Model

#####                     默认测试`config.yaml`中`work_dir`参数对应的文件夹下保存的所有模型在验证集上的效果。如果  `deploy.onnx`和`deploy.tensorrt`均设置为True，在验证集上指标最高的模型将进行模型转换**。**

#####                     首先将`config.yaml`中的`mode`参数设置为`evaluate`，然后运行:

#####                     `python main.py --config_path <config_path>`



###        Run tensorrt model

#####                     基于`tensorrt`进行模型推理，首先将`config.yaml`中的`mode`参数设置为`tensorrt`，运行:

#####                     `python main.py --config_path <config_path>`



###        Demo

#####                     测试单张图片在某个模型上的效果。

#####                     `python demo.py  --config_path <config_path> --model_path <model_path> --image_path  <image_path>`



###        Review

#####                     查看训练模型的特定网络层的类别激活图，具体的网络层及输入图片在`config.yaml`中配置。

#####                     `cd review`

#####                     `python review.py --config_path <config_path>`



​                   

​      

