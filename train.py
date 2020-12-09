"""
Training code
"""

import time
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from load_data import images, labels
from models import ANN, CNN_Basic, CNN

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images = images / abs(images).max() # normalize to [-1, 1]
imagesT = torch.from_numpy(images).float().to(device)
labelsT = torch.from_numpy(labels).long().to(device)
dataset = TensorDataset(imagesT, labelsT)

total_data = len(dataset)
train_split = math.floor(total_data * 0.6)
val_split = math.floor(total_data * 0.2)
test_split = math.ceil(total_data * 0.2)
train_data, val_data, test_data = random_split(dataset, [train_split, val_split, test_split])

def get_accuracy(model, data, batch_size=1):
    correct = 0
    total = 0

    for feature, target in DataLoader(data, batch_size=batch_size):
        if torch.cuda.is_available():
            feature = feature.cuda()
            target = target.cuda()
        feature = feature.unsqueeze(1)
        output = model(feature)
        pred = output.max(1, keepdim=True)[1] # get the index of the max logit
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += feature.shape[0]

    return correct / total

def plot_training_curve(iters, losses, train_acc, val_acc):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    plt.subplot(1,2,2)
    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Train")
    plt.plot(iters, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))
    plt.show()

def train(model, train_data, val_data, batch_size=1, learning_rate=0.01, num_epochs=1):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) # shuffle after every epoch
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    iters, losses, train_acc, val_acc = [], [], [], []

    # training
    print("Starting training")
    n = 0 # the number of iterations
    start_time = time.time()
    for epoch in range(num_epochs):
        for feature, target in iter(train_loader):
            if torch.cuda.is_available():
                feature = feature.cuda()
                target = target.cuda()
            feature = feature.unsqueeze(1)
            out = model(feature)          # forward pass
            loss = criterion(out, target) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

            # save the current training information
            if n % 10 == 9:
                iters.append(n)
                losses.append(float(loss)/batch_size)                         # compute average loss
                train_acc.append(get_accuracy(model, train_data, batch_size)) # compute training accuracy
                val_acc.append(get_accuracy(model, val_data, batch_size))     # compute validation accuracy

            n += 1
        print("Epoch {:2} finished. Time taken: {:6.2f} s".format( epoch, (time.time() - start_time) / (epoch + 1) ))
    end_time = time.time()
    print("Total time: {:6.2f} s  Avg time per Epoch: {:6.2f} s ".format( (end_time - start_time), ((end_time - start_time) / num_epochs) ))
    plot_training_curve(iters, losses, train_acc, val_acc)

# ann = ANN()
# ann.to(device)
# train(ann, train_data, val_data, batch_size=64, num_epochs=100)

# cnn_basic = CNN_Basic()
# cnn_basic.to(device)
# train(cnn_basic, train_data, val_data, batch_size=64, num_epochs=100)

cnn = CNN()
cnn.to(device)
train(cnn, train_data, val_data, batch_size=32, num_epochs=90)
