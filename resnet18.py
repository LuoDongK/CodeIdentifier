from torch.utils.data import Dataset, DataLoader
from torch import optim, nn
from torch.nn import functional as F
import torch
import pandas as pd
import numpy as np


def to_one_hot(code):
    alphabet = [str(ii) for ii in range(0, 10)]
    alphabet += [chr(ii) for ii in range(97, 97 + 26)]
    vec = [0] * 144
    alphabet = ''.join(alphabet)
    for index, char in enumerate(code):
        vec[index * 36 + alphabet.find(char)] = 1
    return vec


class ImageLoader(Dataset):
    def __init__(self, images, labels):
        super(ImageLoader, self).__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img, label = np.array(self.images.iloc[item]), self.labels.iloc[item]
        img = img.reshape((60, 160))
        img = torch.from_numpy(img)
        label = to_one_hot(label)
        label = torch.tensor(label)
        return img, label


class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=2):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes, kernel_size=3):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=kernel_size, stride=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResBlk(64, 128, stride=3)
        self.layer2 = ResBlk(128, 256, stride=3)
        self.layer3 = ResBlk(256, 512, stride=2)
        self.layer4 = ResBlk(512, 1024, stride=2)

        self.out_layer = nn.Linear(1024 * 1 * 2, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.out_layer(x)

        return x


def accuracy(output, target):
    correct_list = []
    output = nn.functional.softmax(output, dim=1)
    output, target = torch.argmax(output, dim=1), torch.argmax(target, dim=1)
    output, target = output.view(-1, 4), target.view(-1, 4)
    for ii, jj in zip(target, output):
        if torch.equal(ii, jj):
            correct_list.append(1)
        else:
            correct_list.append(0)
    acc = sum(correct_list) / len(correct_list)
    return acc


class CodeIdentifier(object):
    def __init__(self, model):
        torch.manual_seed(123)
        self.criterion = nn.MultiLabelSoftMarginLoss()
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.step, self.best_acc, self.best_epoch = 0, 0, 0
        self.validate_accuracy = []
        self.predictions = []

    def train(self, image, label):
        self.model.train()
        image = torch.tensor(image.view(-1, 1, 60, 160), dtype=torch.float)
        image = image.to(device)
        label = label.to(device)
        output = self.model(image)
        loss = self.criterion(output, label)

        with open('train_loss.txt', 'a') as train_loss:
            train_loss.write(str(loss.item()))
            train_loss.write(' ')
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def validate(self, image, label):
        self.model.eval()
        image = torch.tensor(image.view(-1, 1, 60, 160), dtype=torch.float)
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            output = self.model(image).view(-1, 36)

        target = label.view(-1, 36)
        acc = accuracy(output, target)
        self.validate_accuracy.append(acc)
        with open('validate_accuracy.txt', 'a') as validate_accuracy:
            validate_accuracy.write(str(acc))
            validate_accuracy.write(' ')


device = torch.device('cuda')
res = ResNet(144).to(device)
identifier = CodeIdentifier(res)

for epoch in range(120):
    identifier.validate_accuracy = []
    train_iter = pd.read_csv('train.csv', chunksize=200, header=None)
    validate_iter = pd.read_csv('validate.csv', chunksize=200, header=None)
    print('Epoch {} is processing'.format(epoch))
    for chunk_index, chunk in enumerate(train_iter):
        if chunk_index % 50 == 0:
            print('training...')

        label = chunk.iloc[:, -1]
        image = chunk.iloc[:, :-1]
        train_set = ImageLoader(image, label)
        train_loader = DataLoader(train_set, batch_size=200, shuffle=True)
        for x, y in train_loader:
            identifier.train(x, y)

    for chunk_index, chunk in enumerate(validate_iter):
        if chunk_index % 10 == 0:
            print('validating...')

        label = chunk.iloc[:, -1]
        image = chunk.iloc[:, :-1]
        validate_set = ImageLoader(image, label)
        validate_loader = DataLoader(validate_set, batch_size=200, shuffle=True)
        for x, y in validate_loader:
            identifier.validate(x, y)
    avg_acc = np.array(identifier.validate_accuracy).mean()
    if avg_acc > identifier.best_acc:
        identifier.best_epoch = epoch
        identifier.best_acc = avg_acc
        torch.save(identifier.model.state_dict(), 'best.mdl')
    print('第{}代准确率:{}'.format(epoch, avg_acc))
print('The best accuracy {} was obtained in epoch {}'.format(identifier.best_acc, identifier.best_epoch))
