# coding=UTF-8
import numpy as np
from glob import *
import os
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.modules
from torch import optim
from torch.optim import lr_scheduler
import time
from torch.autograd import Variable
from PIL import Image
import sys


def createDataset():
    ##创建数据集
    path = 'myCatAndDotSet/'

    files = glob(os.path.join(path, '*.jpg'))
    print(f'total num of imags :{len(files)}')
    numImages = len(files)

    shuffle = np.random.permutation(numImages)
    os.mkdir(os.path.join(path, 'train'))
    os.mkdir(os.path.join(path, 'valid'))

    #创建目录
    for t in ['valid', 'train']:
        for folder in ['dog', 'cat']:
            os.mkdir(os.path.join(path, t, folder))

    for i in shuffle[:500]:
        folder = files[i].split('/')[-1].split('.')[0]
        image = files[i].split('/')[-1]
        os.rename(files[i], os.path.join(path, 'train', folder, image))

    for i in shuffle[500:1000]:
        folder = files[i].split('/')[-1].split('.')[0]
        image = files[i].split('/')[-1]
        os.rename(files[i], os.path.join(path, 'valid', folder, image))


def loadDataSet():
    simpleTransform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train = ImageFolder('myCatAndDotSet/train', simpleTransform)
    valid = ImageFolder('myCatAndDotSet/valid', simpleTransform)

    # imshow(train[30][0])

    #训练集，一定要用shuffle=True打乱, 否则训练没有效果，原因未知
    trainDataGen = torch.utils.data.DataLoader(train, shuffle=True, batch_size=10, num_workers=1)
    validDataGen = torch.utils.data.DataLoader(valid, batch_size=10, num_workers=1)
    print("dataSize:", len(trainDataGen.dataset))

    dataset_size = {'train': len(trainDataGen.dataset), 'valid': len(validDataGen.dataset)}
    dataloaders = {'train': trainDataGen, 'valid': validDataGen}
    return dataloaders, dataset_size


def imshow(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)


def defineResnet():
    #认识resnet
    resnet = models.resnet18(pretrained=True)
    print(resnet.fc)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 2)

    is_cuda = True
    if is_cuda:
        model_ft = resnet.cuda()
    # print(resnet)
    print(resnet.fc)

    return model_ft


is_cuda = True


#模型训练
def trainingModel():
    DataLoader, dataSize = loadDataSet()

    resnetModel = defineResnet()

    learning_rate = 0.001
    totoalEpoch = 5

    #交叉熵损失
    criterion_loss = nn.CrossEntropyLoss()

    #优化器
    optimizer_ft = optim.SGD(resnetModel.parameters(), lr=learning_rate, momentum=0.9)

    #使用StopLr函数来动态修改学习率
    scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    since = time.time()

    #获取预训练权重数据
    best_model_wts = resnetModel.state_dict()
    bast_acc = 0.0

    #周期为500次
    for epoch in range(totoalEpoch):
        print('周期 {}/{}'.format(epoch, totoalEpoch))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                #设置为训练模式
                scheduler.step()
                resnetModel.train(True)
                print("-------train-start----------")
            else:
                resnetModel.train(False)
                print('----------valid- start--------')

            running_loss = 0.0
            running_corrents = 0

            for data in DataLoader[phase]:
                inputs, labels = data

                if is_cuda:
                    #数据导入到GPU
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                #梯度清0
                optimizer_ft.zero_grad()

                #数据输入模型得到输出
                outputs = resnetModel(inputs)

                #计算预测值 pred是预测值
                _, preds = torch.max(outputs.data, 1)
                # print("pred :", preds)

                #求损失值
                loss = criterion_loss(outputs, labels)
                # print("loss:", loss)

                if phase == 'train':
                    #求梯度
                    loss.backward()
                    # print("backward loss:", loss)
                    #优化权重，更新权重参数， 反向传播
                    optimizer_ft.step()

                # running_loss += loss.data[0]              #新版本不是这样用，用法如下
                running_loss += loss.item()
                running_corrents += torch.sum(preds == labels.data)
                # print('preds:', preds)
                # print('labels.data:', labels.data)
                # print('preds == labels.data:', preds == labels.data)
                # print('runnnig loss:', loss)
                # print('running_corrents:', running_corrents)
                # print('datasize:', dataSize[phase])

            #原来的tensoer 与整数之间的/ 被废弃 改用如下方式
            epoch_loss = torch.true_divide(running_loss, dataSize[phase])
            epoch_acc = torch.true_divide(running_corrents, dataSize[phase])

            # print('{} Loss:{:.4f} Acc:{:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('epoch_loss:', epoch_loss)
            # print('epoch_acc:', epoch_acc)

            if phase == 'valid' and epoch_acc > bast_acc:
                #acc 是准确率
                bast_acc = epoch_acc
                best_model_wts = resnetModel.state_dict()
            print()
        time_elapsed = time.time() - since
        print("trainning complete in {:.0f}m {:.0f}s".format(time_elapsed / 60, time_elapsed % 60))
        print('Best val Acc:{:.4f}'.format(bast_acc))

        time.sleep(1)
    # 加载训练好的模型参数
    resnetModel.load_state_dict(best_model_wts)

    # img = Image.open('dog.2.jpg').convert('RGB')
    # img = simpleTransform(img).unsqueeze(0)
    #
    # img = img.to(device, torch.float)
    #
    # outputs = resnetModel(img)
    # print('output:', outputs)
    # _,preds =torch.max(outputs.data, 1)
    #
    # print('preds:',preds)
    # print("图像检测结果为:",class_label[preds[0]])

    #最终保存模型
    torch.save(resnetModel, "1.pth")  #保存整个网络和参数
    # torch.save(resnetModel.state_dict(), "1.pth")       #只保存参数（未成功）


class_label = ["cat", "dog"]
model_path = '1.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
simpleTransform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


#测试模型
def useTrainedModelDetect(imgName):
    # 只加载保存的参数（未成功）
    # resnetModel = defineResnet()
    # params = torch.load(model_path, device)
    # resnetModel.load_state_dict(params)

    print(sys.argv[0], sys.argv[1])

    # 加载保存的整个网络和参数
    resnetModel = torch.load(model_path)

    img = Image.open(imgName).convert('RGB')
    img = simpleTransform(img).unsqueeze(0)

    img = img.to(device, torch.float)

    outputs = resnetModel(img)
    print('output:', outputs)
    _, preds = torch.max(outputs.data, 1)

    print('preds:', preds)
    print("图像检测结果为:", class_label[preds[0]])


if __name__ == "__main__":
    # createDataset()
    # imageTransforms()
    # defineResnet()
    # trainingModel()

    useTrainedModelDetect(sys.argv[1])