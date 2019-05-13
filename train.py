import argparse

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


parser = argparse.ArgumentParser(description='Training Model')

parser.add_argument('data_directory', action = 'store',help = 'Path for traning data')

parser.add_argument('--arch', action='store',dest = 'pretrained_model', default = 'vgg19',help= 'Enter Model name in which data will be trained.')

parser.add_argument('--save_dir', action = 'store',
                    dest = 'save_directory', default = 'checkpoint2.pth',
                    help = 'Path for Checkpoint')

parser.add_argument('--learning_rate', action = 'store',
                    dest = 'lr', type=int, default = 0.003,
                    help = 'Learing rate for model')


parser.add_argument('--hidden_units', action = 'store',
                    dest = 'hidden_layers', type=int, default = 512,
                    help = 'Number of hidden units.')

parser.add_argument('--epochs', action = 'store',
                    dest = 'epochs', type = int, default = 5,
                    help = 'Number of epochs for training')

parser.add_argument('--gpu', action = "store_true", default = False,
                    help = 'Turn GPU mode on or off.')

results = parser.parse_args()

data_dir = results.data_directory
save_dir = results.save_directory
learning_rate = results.lr
arch = results.pretrained_model.lower()
hidden_layers = results.hidden_layers
epochs = results.epochs
gpu_mode = results.gpu
                    
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)


trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True)
vldLoader = torch.utils.data.DataLoader(validation_data, batch_size =64,shuffle = True)



# Use GPU if it's available
device = torch.device("cpu")
if gpu_mode:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
if arch == 'alexnet':
    model = models.alexnet(pretrained=True)
    input_units = 9216
elif arch == 'vgg19':
    model = models.vgg19(pretrained=True)
    input_units = 25088
elif arch == 'densenet121':
    model = models.densenet121(pretrained=True)
    input_units = 1024
else:
    print('Model not recongized.')
    sys.exit()


for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(input_units, hidden_layers),
                                 nn.ReLU(),
                                 nn.Linear(hidden_layers,512),
                                 nn.ReLU(),
                                 nn.Linear(512,256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()


optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

model.to(device);


steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            validation_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in vldLoader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    validation_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {validation_loss/len(vldLoader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(vldLoader):.3f}")
            running_loss = 0
            model.train()
            
def test_accuracy(testloader):    
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Network Accuracy on the test data: %d %%' % (100 * correct / total))
    
    
test_accuracy(testloader)
            
model.class_to_idx = train_data.class_to_idx

model.cpu
torch.save({'structure' :arch,
            'input_size' : input_units,
            'hidden_layer1':hidden_layers,
            'output_size' : 102,
            'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx},
            save_dir)


print("Training is complete")