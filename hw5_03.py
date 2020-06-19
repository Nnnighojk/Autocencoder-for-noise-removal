""" 
----------------Output-----------------------
Noise Classification Accuracy: 99.05000000000001
Noise Confusion Matrix:
                       no-noise       20-noise       50-noise       80-noise

    no-noise:            98.00           1.90           0.10           0.00
    20-noise:             0.00          99.90           0.10           0.00
    50-noise:             0.00           0.00          98.30           1.70
    80-noise:             0.00           0.00           0.00         100.00
Dataset0 Classification Accuracy: 77.25
Dataset0 Confusion Matrix:
                      rectangle       triangle           disk           oval           star

   rectangle:            52.88           0.62           6.88          32.00           7.62
    triangle:             4.25          82.00          10.62           0.62           2.50
        disk:             3.75           0.00          91.00           1.62           3.62
        oval:            22.50           0.12           4.38          71.12           1.88
        star:             5.62           0.00           3.25           1.88          89.25
Dataset20 Classification Accuracy: 77.47500000000001
Dataset20 Confusion Matrix:
                      rectangle       triangle           disk           oval           star

   rectangle:            54.12           0.50           6.75          32.00           6.62
    triangle:             4.75          82.62          10.25           0.25           2.12
        disk:             3.38           0.00          91.62           1.88           3.12
        oval:            24.25           0.12           3.62          70.25           1.75
        star:             5.75           0.00           3.25           2.25          88.75
Dataset50 Classification Accuracy: 77.14999999999999
Dataset50 Confusion Matrix:
                      rectangle       triangle           disk           oval           star

   rectangle:            53.75           0.50           6.75          32.25           6.75
    triangle:             4.62          82.12          10.38           0.50           2.38
        disk:             3.62           0.12          90.62           2.00           3.62
        oval:            23.50           0.12           4.12          70.25           2.00
        star:             6.12           0.00           2.88           2.00          89.00
Dataset80 Classification Accuracy: 77.225
Dataset80 Confusion Matrix:
                      rectangle       triangle           disk           oval           star

   rectangle:            53.75           0.62           7.25          31.75           6.62
    triangle:             4.62          82.25          10.50           0.12           2.50
        disk:             3.62           0.00          91.75           1.25           3.38
        oval:            24.00           0.00           3.62          70.12           2.25
        star:             6.38           0.00           3.00           2.38          88.25
"""
from DLStudio import *
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#import gzip, pickle
#def getCombinedResult(file):
#    final = list()
#
#    final.append(dict())
#    #final.append( {'oval': 3, 'star': 4, 'disk': 2, 'triangle': 1, 'rectangle': 0})
#    final.append({'no-noise':0,'20-noise':1,'50-noise':2,'80-noise':3})
#    index = 0
#    for i in range(len(file)):
#        filepath = "DLStudio-1.1.0/Examples/data/"+file[i]
#        f = gzip.open(filepath, 'rb')
#        dataset =f.read()
#        output = pickle.loads(dataset, encoding='latin1')
#        for k,v in output[0].items():
#            v.append(i)
#            final[0][index] = v
#            index+=1
#    return final
#
#train_dataset = getCombinedResult(file =['PurdueShapes5-10000-train.gz','PurdueShapes5-10000-train-noise-20.gz','PurdueShapes5-10000-train-noise-50.gz','PurdueShapes5-10000-train-noise-80.gz'])
#with gzip.open('DLStudio-1.1.0/Examples/data/train.gz', 'wb') as f:
#    f.write(pickle.dumps(train_dataset))
#test_dataset = getCombinedResult(file =['PurdueShapes5-1000-test.gz','PurdueShapes5-1000-test-noise-20.gz','PurdueShapes5-1000-test-noise-50.gz','PurdueShapes5-1000-test-noise-80.gz'])
#with gzip.open('DLStudio-1.1.0/Examples/data/test.gz', 'wb') as f:
#    f.write(pickle.dumps(test_dataset))


from DLStudio import DLStudio


class OverloadedPurdueShapes5Dataset(DLStudio.DetectAndLocalize.PurdueShapes5Dataset):
    def __getitem__(self, idx):
        r = np.array( self.dataset[idx][0] )
        g = np.array( self.dataset[idx][1] )
        b = np.array( self.dataset[idx][2] )
        R,G,B = r.reshape(32,32), g.reshape(32,32), b.reshape(32,32)
        im_tensor = torch.zeros(3,32,32, dtype=torch.float)
        im_tensor[0,:,:] = torch.from_numpy(R)
        im_tensor[1,:,:] = torch.from_numpy(G)
        im_tensor[2,:,:] = torch.from_numpy(B)
        bb_tensor = torch.tensor(self.dataset[idx][3], dtype=torch.float)
        sample = {'image' : im_tensor, 
                'bbox' : bb_tensor,
                'label' : self.dataset[idx][4] }
        if len(self.dataset[idx]) == 6:
            sample['noise'] = self.dataset[idx][5]
        if self.transform:
            sample = self.transform(sample)
        return sample


def smoothen_image(dataset, dataset_id):
    kernel = np.array([[1,0,-1],[0,0,0],[-1,0,1]], dtype=np.uint8)
    output_dataset = []
    for idx in range(len(dataset)):
        output_dataset.append([])
        r = np.array( dataset[idx][0] )
        g = np.array( dataset[idx][1] )
        b = np.array( dataset[idx][2] )
        R,G,B = r.reshape(32,32), g.reshape(32,32), b.reshape(32,32)
        im_tensor = torch.zeros(3,32,32, dtype=torch.float)
        im_tensor[0,:,:] = torch.from_numpy(R)
        im_tensor[1,:,:] = torch.from_numpy(G)
        im_tensor[2,:,:] = torch.from_numpy(B)
        imt = im_tensor/255
        imt_tmp = imt.numpy().transpose(1,2,0)
        if dataset_id in [50, 80, "full"]:
            imt_new = cv2.bilateralFilter(imt_tmp, 9, 75, 75).transpose(2,0,1)
        elif dataset_id == 20:
            imt_new = cv2.GaussianBlur(imt_tmp, (5,5), 0).transpose(2,0,1)

        imt_new = torch.from_numpy(imt_new)
        imt_new_tensor = imt_new * 255
        R_new = imt_new_tensor[0, :, :]
        G_new = imt_new_tensor[1, :, :]
        B_new = imt_new_tensor[2, :, :]
        r_new, g_new, b_new = R_new.reshape(1024), G_new.reshape(1024), B_new.reshape(1024)
        output_dataset[idx].append(r_new)
        output_dataset[idx].append(g_new)
        output_dataset[idx].append(b_new)
        output_dataset[idx].append(dataset[idx][3])
        output_dataset[idx].append(dataset[idx][4])
    return output_dataset


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(3, 128, 3, stride=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(128, 64, 3, stride=2, padding=1), 
                nn.ReLU(),
                nn.MaxPool2d(2, stride=1)
                )
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 128, 3, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 5, stride=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, 2, stride=2, padding=1),
                nn.ReLU()
                )
        self.fc1 = nn.Linear(2352, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 2352)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Task2Net(nn.Module):
    def __init__(self, depth=16, noise=20):
        super(Task2Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64,  3)
        self.conv2 = nn.Conv2d(64, 128,  3)
        self.pool = nn.MaxPool2d(2, 2)
        self.b1 = nn.BatchNorm2d(128)
        self.b2 = nn.BatchNorm2d(256)
        self.noise = noise
        self.depth = depth
        self.mod_list = nn.ModuleList()
        for i in range(self.depth * noise // 5):
            self.mod_list.append(DLStudio.DetectAndLocalize.SkipBlock(128, 128, skip_connections=True))
        self.skip128ds = DLStudio.DetectAndLocalize.SkipBlock(128, 128, downsample=True,
                skip_connections=True)
        self.skip128to256 = DLStudio.DetectAndLocalize.SkipBlock(128, 256, 
                skip_connections=True)
        self.fc1 =  nn.Linear(2304, 1000)
        self.fc2 =  nn.Linear(1000, 5)


    def forward(self, x):
        # Classification
        x1 = self.pool(torch.nn.functional.relu(self.conv1(x))) 
        x1 = self.pool(torch.nn.functional.relu(self.conv2(x1)))
        for _, skip128 in enumerate(self.mod_list[:self.depth*self.noise//4]):
            x1 = skip128(x1)
        x1 = self.skip128ds(x1)
        for _, skip128 in enumerate(self.mod_list[self.depth*self.noise//4:]):
            x1 = skip128(x1)
        x1 = self.b1(x1)
        x1 = self.skip128to256(x1)
        x1 = self.b2(x1)
        x1 = x1.view(-1, 2304 )
        x1 = torch.nn.functional.relu(self.fc1(x1))
        x1 = self.fc2(x1)

        return x1


def run_code_for_training(net, lr):
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    epochs = 1
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(detector.train_dataloader):
            inputs, _, labels = data['image'], data['bbox'], data['noise']
            inputs = inputs.to(device,non_blocking=True)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

def run_code_for_training1(net, lr):
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    epochs = 1
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(detector.train_dataloader):
            inputs, _, labels = data['image'], data['bbox'], data['label']
            inputs = inputs.to(device,non_blocking=True)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

def run_code_for_testing(net):
    confusion_matrix = torch.zeros(5,5)
    test_correct = 0
    final_dataset = {}
    class_total = [0] * len(dataserver_train.class_labels)
    for i, data in enumerate(detector.test_dataloader):
        inputs, _, labels = data['image'], data['bbox'], data['noise']
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        for label,prediction in zip(labels,predicted):
            final_labels = prediction.data.item()          
            if not final_dataset.get(final_labels):               
                final_dataset[final_labels] = list()
            final_dataset[final_labels].append(data)
            confusion_matrix[label][prediction] += 1
        test_correct += sum([1 for i in range(len(predicted)) if predicted[i] == labels[i]])
        for j in range(4):
            label = labels[j]
            class_total[label] += 1
    test_acc = 100 * (test_correct / len(detector.test_dataloader.dataset))
    print("Noise Classification Accuracy: {0}".format(test_acc))
    print("Noise Confusion Matrix:")
    out_str = "                "
    for j in range(len(dataserver_train.class_labels)):
        out_str +=  "%15s" % dataserver_train.class_labels[j]
    print(out_str + "\n")
    for i,label in enumerate(dataserver_train.class_labels):
        out_percents = [100 * confusion_matrix[i,j] / float(class_total[i])
                for j in range(len(dataserver_train.class_labels))]
        out_percents = ["%.2f" % item.item() for item in out_percents]
        out_str = "%12s:  " % dataserver_train.class_labels[i]
        for j in range(len(dataserver_train.class_labels)):
            out_str +=  "%15s" % out_percents[j]
        print(out_str)
    return final_dataset

def run_code_for_testing1(net, k):
    confusion_matrix = torch.zeros(5,5)
    test_correct = 0
    final_dataset = {}
    class_total = [0] * 5
    for i, data in enumerate(detector.test_dataloader):
        inputs, _, labels = data['image'], data['bbox'], data['label']
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        for label,prediction in zip(labels,predicted):
            confusion_matrix[label][prediction] += 1
        test_correct += sum([1 for i in range(len(predicted)) if predicted[i] == labels[i]])
        for j in range(4):
            label = labels[j]
            class_total[label] += 1
    test_acc = 100 * (test_correct / len(detector.test_dataloader.dataset))
    print("Dataset{0} Classification Accuracy: {1}".format(k, test_acc))
    print("Dataset{0} Confusion Matrix:".format(k))
    out_str = "                "
    for j in range(len(dataserver_train.class_labels)):
        out_str +=  "%15s" % dataserver_train.class_labels[j]
    print(out_str + "\n")
    for i,label in enumerate(dataserver_train.class_labels):
        out_percents = [100 * confusion_matrix[i,j] / float(class_total[i])
                for j in range(len(dataserver_train.class_labels))]
        out_percents = ["%.2f" % item.item() for item in out_percents]
        out_str = "%12s:  " % dataserver_train.class_labels[i]
        for j in range(len(dataserver_train.class_labels)):
            out_str +=  "%15s" % out_percents[j]
        print(out_str)


if __name__=='__main__':
    datasets = {
            "full": ["train.gz", "test.gz"],
            0: ["PurdueShapes5-10000-train.gz",
                "PurdueShapes5-1000-test.gz"],
            20: ["PurdueShapes5-10000-train-noise-20.gz",
                "PurdueShapes5-1000-test-noise-20.gz"],
            50: ["PurdueShapes5-10000-train-noise-50.gz",
                "PurdueShapes5-1000-test-noise-50.gz"],
            80: ["PurdueShapes5-10000-train-noise-80.gz",
                "PurdueShapes5-1000-test-noise-80.gz"],
            }
    lr = 1e-4

    dls = DLStudio(
            dataroot = "DLStudio-1.1.0/Examples/data/",
            image_size = [32,32],
            path_saved_model = "./saved_model",
            momentum = 0.9,
            learning_rate = lr,
            epochs = 2,
            batch_size = 4,
            classes = ('no-noise','20-noise','50-noise','80-noise'),
            debug_train = 0,
            debug_test = 0,
            use_gpu = True,
            )
    detector = DLStudio.DetectAndLocalize( dl_studio = dls )
    dataserver_train = OverloadedPurdueShapes5Dataset(
            train_or_test = 'train',
            dl_studio = dls,
            dataset_file = datasets["full"][0])

    dataserver_test = OverloadedPurdueShapes5Dataset(
            train_or_test = 'test',
            dl_studio = dls,
            dataset_file = datasets["full"][1])


    detector.dataserver_train = dataserver_train
    detector.dataserver_test = dataserver_test
    detector.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
            batch_size=4,shuffle=True,
            num_workers=0)
    detector.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
            batch_size=4,shuffle=True,
            num_workers=0)

    model = autoencoder()
    run_code_for_training(model, lr)
    final_dataset = run_code_for_testing(model)


    dls = DLStudio(
            dataroot = "DLStudio-1.1.0/Examples/data/",
            image_size = [32,32],
            path_saved_model = "./saved_model",
            momentum = 0.9,
            learning_rate = lr,
            epochs = 2,
            batch_size = 4,
            classes = ('rectangle','triangle','disk','oval','star'),
            debug_train = 0,
            debug_test = 0,
            use_gpu = True,
            )
    detector = DLStudio.DetectAndLocalize( dl_studio = dls )
    dataserver_train = OverloadedPurdueShapes5Dataset(
            train_or_test = 'train',
            dl_studio = dls,
            dataset_file = datasets["full"][0])
    dataserver_test = OverloadedPurdueShapes5Dataset(
            train_or_test = 'test',
            dl_studio = dls,
            dataset_file = datasets["full"][1])

    detector.dataserver_train = dataserver_train
    class_labels = {'oval': 3, 'star': 4, 'disk': 2, 'triangle': 1, 'rectangle': 0}
    detector.dataserver_train.class_labels = dict(map(reversed, class_labels.items()))
    detector.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
            batch_size=4,shuffle=True,
            num_workers=0)

    #model = detector.LOADnet2(skip_connections=True, depth=32)

    for k,v in enumerate([0, 20, 50, 80]):
        model = Task2Net(depth=32, noise=v)
        run_code_for_training1(model, lr)

        detector.dataserver_test = final_dataset[k]
        detector.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                batch_size=4,shuffle=True,
                num_workers=0)

        run_code_for_testing1(model, v)


