import torch
import torch.nn as nn
import torch.optim as optim
from bus_presence import MediumNet
from dataset import FeatureMapDataset, extract_features
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as tf
import glob
import os

def labels(file="coral-reef-long.csv"):

    label_file = "/home/workstation1/yolov2_big_files/video/coral-reef-long/coral"
    train_imgs_path = "/home/workstation1/yolov2_big_files/video/coral-reef-long/train/*.jpg"
    names = [os.path.basename(x) for x in glob.glob(train_imgs_path)]
    path = "/home/workstation1/yolov2_big_files/video/coral-reef-long/"
    t_labels = dict()
    temp = list()
    iter = 0
    with open(path + file, "r") as f:
        for i, row1 in enumerate(f):
            if i > 0:
                row = row1.strip().split(",")
                if "person" == row[1] and row[0] not in temp and row[0] in names:
                    t_labels[iter] = 1.0
                    iter += 1
                    temp.append(row[0])
    print(len(t_labels))
    torch.save(t_labels, label_file)
    return t_labels


def preprocessing():
    extract_features("/home/workstation1/Downloads/bdd100k/images/10k/test1/test/r")


def training():
    label_file = "/home/workstation1/yolov2_big_files/video/DETRAC-sequences/concept_drift_detrac/labels/labels40851"

    labels = torch.load(label_file)
    net = MediumNet(3)
    model_name = "bus_presence_MVI_40851235_"
    model_path = "/home/workstation1/yolov2_big_files/video/DETRAC-sequences/concept_drift_detrac/"

    dataset = FeatureMapDataset("/home/workstation1/yolov2_big_files/video/DETRAC-sequences/" +
                                "concept_drift_detrac/seq2/features40851/",
                                label=labels, flip=False)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = net.cuda()
    num_epochs = 50
    learning_rate = 1e-3
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    for epoch in range(num_epochs):
        for ix, d in enumerate(dataloader):
            im, label = d
            im = im.cuda()
            optimizer.zero_grad()
            output = model(im)
            label = label.type(torch.cuda.FloatTensor)
            x_cuda = Variable(label, requires_grad=False).cuda()
            output_cuda = Variable(output, requires_grad=True).cuda()
            loss = criterion(output_cuda, x_cuda)
            print(ix, loss)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        print('epoch [{}/{}], loss:{:.3f}'.format(epoch + 1, num_epochs, loss.item()))
        if epoch == 10 or epoch == 20 or epoch == 30 or epoch == 40:
            torch.save(model.state_dict(), model_path + model_name + str(epoch) + '.pth')
    torch.save(model.state_dict(), model_path + model_name + "50.pth")


preprocessing()
