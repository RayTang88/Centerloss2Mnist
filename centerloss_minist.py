import os
import time
import datetime
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import nn
from torch import optim as optim
from matplotlib import pyplot as plt

class LeNets(nn.Module):
    def __init__(self):
        super(LeNets, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.PReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.PReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.PReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
            )

        self.liner1 = nn.Sequential(
            nn.Linear(in_features=128 * 3 * 3,
                      out_features=2),
            nn.PReLU()
        )
        self.liner2 = nn.Linear(in_features=2,
                                out_features=10)

    def forward(self,input):
        x = self.conv1(input)
        x = x.view(-1, 128*3*3)
        Coordinate = self.liner1(x)
        Predict = self.liner2(Coordinate)
        # F.log_softmax(Predict, dim=1)

        return Coordinate, F.log_softmax(Predict, dim=1)

class Centerloss(nn.Module):
    def __init__(self, class_num, feat_num, iscuda):
        super(Centerloss, self).__init__()
        self.iscuda = iscuda
        self.center = nn.Parameter(torch.randn(class_num, feat_num))
        if self.iscuda:
            self.center.cuda()

    def forward(self, coordinate, labels):

        labels = labels.cpu().float()
        count = torch.histc(labels, 10, min=0, max=9).cuda()
        labels = labels.cuda()
        num = torch.index_select(count, 0, labels.long())
        centers = torch.index_select(self.center, 0, labels.long())
        loss = torch.sum(torch.sqrt(torch.sum((coordinate - centers)**2, dim=1))/num)/labels.size(0)
        return loss

class Visualization:
    def __init__(self, coordinates, labels, epoch, save_path):
        self.c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
             '#ff00ff', '#990000', '#999900', '#009900', '#009999']
        self.coordinates = coordinates
        self.labels = labels
        self.epoch = epoch
        self.save_path = save_path
        self.forward()

    def forward(self):
        plt.ion()
        plt.clf()

        for i in range(10):
            plt.title('Centerloss')
            plt.plot(self.coordinates[self.labels == i, 0], self.coordinates[self.labels == i, 1], '.', color=self.c[i])
            plt.xlim(left=-5, right=5)
            plt.ylim(bottom=-5, top=5)
            plt.text(-4, 4, 'epoch={}'.format(self.epoch))
            plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
            plt.savefig(os.path.join(self.save_path, 'epoch={}.jpg'.format(self.epoch)))
        plt.show()
        plt.pause(0.1)
        plt.ioff()

class Train:
    def __init__(self, path, softmaxloss_para_path, centerloss_para_path, save_path, lambda_parameters, iscuda):
        self.iscuda = iscuda
        self.lenet = LeNets()
        self.nllloss = nn.NLLLoss()
        # self.nllloss = nn.CrossEntropyLoss()
        self.centerloss = Centerloss(10, 2, self.iscuda)
        self.path = path
        self.save_path = save_path
        self.softmax_para_path = softmaxloss_para_path
        self.centerloss_para_path = centerloss_para_path
        self.lambda_parameters = lambda_parameters

        self.optimizernn = optim.Adam(self.lenet.parameters(), lr=0.0005)
        self.optimizerct = optim.SGD(self.centerloss.parameters(), lr=0.001)

        if os.path.exists(self.path):
            self.lenet.load_state_dict(torch.load(self.softmax_para_path))
            self.centerloss.load_state_dict(torch.load(self.centerloss_para_path))
        if self.iscuda:
            self.lenet.cuda()
            self.centerloss.cuda()
        self.train()

    def train(self):
        coordinates = []
        labels = []
        flag = 1.5

        for i, (data, label) in enumerate(dataloder):
            if self.iscuda:
                data = data.cuda()
                label = label.cuda()
            coordinate, predict = self.lenet(data)

            softmaxloss = self.nllloss(predict, label)
            centerloss = self.centerloss(coordinate, label)
            loss = softmaxloss + self.lambda_parameters * centerloss

            coordinates.append(coordinate)
            labels.append(label)

            if loss < flag:
                if not os.path.exists(self.path):
                    os.mkdir(self.path)
                torch.save(self.lenet.state_dict(), self.softmax_para_path)
                torch.save(self.centerloss.state_dict(), self.centerloss_para_path)
                flag = loss
            self.optimizernn.zero_grad()
            self.optimizerct.zero_grad()
            loss.backward()
            self.optimizernn.step()
            self.optimizerct.step()
            print('训练批次:{}'.format(epoch))
            print('total_loss:', loss.item())
            print('softmaxloss:', softmaxloss.item())
            print('centerlosss:', centerloss.item())

        coord = torch.cat(coordinates).cpu().data.numpy()
        lab = torch.cat(labels).cpu().data.numpy()

        if epoch % 1 == 0:
            Visualization(coord, lab, epoch, self.save_path)

if __name__ == '__main__':
    start_time = time.time()
    path = './parameters8'
    softmaxloss_para_path = './parameters8/Softmaxloss.pkl'
    centerloss_para_path = './parameters8/Centerloss.pkl'
    save_path = './images8'

    lambda_parameters = 1
    epoch = 0

    mydataset = MNIST('./MNIST', train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]), download=True)
    dataloder = DataLoader(mydataset, batch_size=128, shuffle=True, num_workers=4)
    for _ in range(100):
        train = Train(path, softmaxloss_para_path, centerloss_para_path, save_path, lambda_parameters, True)
        epoch += 1


    Train_time = (time.time() - start_time) / 60
    print('{}训练耗时:'.format('centerloss'), int(Train_time), 'minutes')
    print(datetime.datetime.now())
