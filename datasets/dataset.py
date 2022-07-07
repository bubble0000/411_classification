import cv2
from torchvision import datasets
from torch.utils.data import Dataset

class VGGDataset(Dataset):
    def __init__(self, train_path=None,test_path=None,train=True,transform_data=None,transform_data_test=None):
        if train:
            self.img_datas = datasets.ImageFolder(train_path)
            self.transform = transform_data
        else:
            self.img_datas = datasets.ImageFolder(test_path)
            self.transform = transform_data_test
        self.train = train

    def __getitem__(self, idx):

        imgA = cv2.imread(self.img_datas.imgs[idx][0])
        label = self.img_datas.imgs[idx][1]
        img_name = self.img_datas.imgs[idx][0]
        imgA = self.transform(imgA)
        imgA = imgA.cuda()

        return imgA, label, img_name

    def __len__(self):
        return len(self.img_datas.imgs)


