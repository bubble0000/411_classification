import os
import torch
import time
from tqdm import tqdm
import numpy as np

from core.class_net_train import class_net_train
from core.model_state_dict import densenet_state_dict,model_state_dict
from utils.logger import Logger


class model_train(class_net_train):
    def __init__(self,config,log_path):
        super().__init__(config)

        self.start_epoch = -1
        if self.resume_model():
            print("resume model!")
        elif config.pretrain.load_pretrained:
            self.load_pretrained(self.config.pretrain)

        self.logger = Logger(logname=log_path, logger="Loss").getlog()

    def load_pretrained(self,config):
        model_dict = self.model.state_dict()
        if self.config.model.arch.find('densenet') >= 0:
            state_dict = densenet_state_dict(config)
        else:
            state_dict = model_state_dict(model_dict,config)

        self.model.load_state_dict(state_dict,strict=False)
        self.model.cuda()

    def resume_model(self):
        if self.config.resume:
            models_path = os.path.join(self.config.work_dir,'models')
            models = os.listdir(models_path)
            models.sort(key=lambda x:int(x[:-4]))

            if len(models):
                latest_checkpoint_path = os.path.join(models_path,models[-1])
                print("resume " + str(latest_checkpoint_path))
                checkpoint = torch.load(latest_checkpoint_path)
                self.model.load_state_dict(checkpoint['net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                self.start_epoch = checkpoint['epoch']
                return True
        return False

    def validate(self):
        self.model.eval()
        image_labels = []
        pred_labels = []

        for i,data in tqdm(enumerate(self.val_dataloader,0)):
            images,labels,image_names = data
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            for l in labels:
                image_labels.append(l.cpu().numpy())
            for pred_l in predicted:
                pred_labels.append(pred_l.cpu().numpy())

        image_labels = np.array(image_labels)
        pred_labels = np.array(pred_labels)
        acc = np.sum(image_labels == pred_labels) / image_labels.shape[0]
        return acc

    def train(self):
        print("start training")
        for epo in range(self.start_epoch + 1,self.config.epoch):
            self.lr_scheduler.step()
            time_start = time.time()
            for i, data in enumerate(self.data_loader, 0):
                self.model.train()
                inputs, y,_ = data

                inputs = torch.autograd.Variable(inputs)
                y = torch.autograd.Variable(y)
                inputs = inputs.cuda()
                y = y.cuda()

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, y)

                loss.backward()
                self.optimizer.step()

                if epo == 0 and i == 0:
                    self.logger.info("Train config: ")
                    for key,val in self.config.items():
                        if isinstance(val,dict):
                            self.logger.info(str(key) + " : ")
                            for _key,_val in val.items():
                                self.logger.info("\t" + str(_key) + " : " + str(_val))
                        else:
                            self.logger.info(str(key) + " : " + str(val))
                self.logger.info("Loss: epoch: " + str(epo) + " iter_num: " + str(i) + " loss: " + str(loss.item()))
                print("Loss: ",str(loss.item()))

            time_end = time.time()
            self.logger.info("epoch training time: " + str(time_end - time_start))

            if epo % self.config.test_peroid == 0:
                acc = self.validate()
                self.logger.info("Val accuracy: " + str(acc))

            if epo % self.config.ckpt_peroid == 0:
                model_save_dir = os.path.join(self.config.work_dir,'models/')
                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)

                checkpoint = {
                    "net": self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    "epoch": epo,
                    'lr_scheduler': self.lr_scheduler.state_dict()
                }

                torch.save(checkpoint,model_save_dir + "%s.pth" % str(epo))

