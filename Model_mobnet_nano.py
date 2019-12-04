import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.models import mobilenet_v2, MobileNetV2
from utils import progress_bar
from datetime import datetime


#data_dir  = '/Irma/train'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('==> Preparing data..')
transform_train = transforms.Compose([transforms.Resize((224,224)),
    #transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
#root = []
train_data = torchvision.datasets.ImageFolder(root='./12-retrain/clean-dataset/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=80, shuffle=True, num_workers=2)

#print(type(trainset.classes))


print(train_data.class_to_idx)
#idx_to_class = {j:i for i,j in trainset.class_to_idx.items()}

test_data = torchvision.datasets.ImageFolder(root='./12-retrain/clean-dataset/validation',  transform=transform_test)
testloader = torch.utils.data.DataLoader(test_data, batch_size=40, shuffle=False, num_workers=2)
print(test_data.class_to_idx)
#trainloader, testloader = load_split_train_test(data_dir, .2)
print(trainloader.dataset.classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Freeze parameters so we don't backprop through them
tstart = datetime.now()


model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
for params in model.parameters():
     params.requires_grad=True
    

#model=mobilenet_v2()
#model = models.mobilenet_v2(pretrained=True)
model


model.classifier[1] = nn.Sequential(nn.Linear(in_features=model.classifier[1].in_features, out_features=512), nn.ReLU(),nn.Linear(in_features=512, out_features=12), nn.Softmax(dim=1))

print(model.classifier)

model.to(device)

for param in model.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)#, momentum=0.9, weight_decay=5e-4)
train_losses, test_losses = [], []
train_accuracy, test_accuracy = [], []
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    #train_losses, test_losses = [], []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        #print('train')
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    train_losses.append(train_loss/(batch_idx+1))
    train_accuracy.append(100.*correct/total) 
            
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            #print(correct)
            
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        test_accuracy.append(100.*correct/total) 
            
        test_losses.append(test_loss/(batch_idx+1))

for epoch in range(start_epoch, start_epoch+500):
    train(epoch)
    test(epoch)
tend = datetime.now()

ep =  [i for i in range(500)]
plt.plot(ep , train_losses, label='Training loss')
plt.plot(ep , test_losses, label='Validation loss')
plt.plot
plt.savefig('./graphs/loss/Loss')
plt.legend(frameon = False)
plt.show()

plt.plot(ep, train_accuracy , label='Training accuracy')
plt.plot(ep,test_accuracy , label='Validation accuracy')
plt.plot
plt.savefig('./graphs/loss/Loss')
plt.legend(frameon = False)
plt.show()
delta = tend - tstart
print('training time is', delta)
torch.save(model.state_dict(), './Mob_classifier_withgraphs.pth')
print('model is saved')


#for param_tensor in model.state_dict():
    #print(param_tensor, "\t", model.state_dict()[param_tensor].size())

#model.load_state_dict(torch.load('./Mob_classifier1.pth'))
#model.eval()

'''torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)
#model = TheModelClass(*args, **kwargs)
#optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load()
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()'''


