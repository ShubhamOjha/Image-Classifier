import argparse

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
import sys
from PIL import Image

parser = argparse.ArgumentParser(description='Training Model')

parser.add_argument('image_path', action = 'store',help = 'Path for Image data')
parser.add_argument('checkpoint', action = 'store',help = 'Checkpoint to load data into Model')

parser.add_argument('--top_k', action='store',dest = 'topk_val', default = 3,help= 'Enter Number of top probabilities to be returned.')

parser.add_argument('--category_names', action = 'store',
                    dest = 'category_names', default = 'cat_to_name.json',help = 'Category Mapping')

parser.add_argument('--gpu', action = "store_true", default = False,
                    help = 'Turn GPU mode on or off.')
results = parser.parse_args()

image_path = results.image_path
checkpoint = results.checkpoint
topk_val = results.topk_val
category_names = results.category_names

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,map_location=lambda storage, loc: storage)
    arch = checkpoint['structure'].lower()
    print(arch)
    if arch == 'alexnet':
        model_chk = models.alexnet(pretrained=True)
    elif arch == 'vgg19':
        model_chk = models.vgg19(pretrained=True)
    elif arch == 'densenet121':
        model_chk = models.densenet121(pretrained=True)
    else:
        print('Model not recongized.')
        sys.exit()
    
    
    model_chk.classifier = nn.Sequential(nn.Linear(checkpoint['input_size'], checkpoint['hidden_layer1']),
                                 nn.ReLU(),
                                 nn.Linear(checkpoint['hidden_layer1'] ,512),
                                 nn.ReLU(),
                                 nn.Linear(512,256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, checkpoint['output_size']),
                                 nn.LogSoftmax(dim=1))
    
    model_chk.class_to_idx = checkpoint['class_to_idx']

    model_chk.load_state_dict(checkpoint['state_dict'])
    
    return model_chk

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
    
            
    return np_image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model_chk.eval()
    img_torch = process_image(image_path)
    img_torch = torch.tensor(img_torch)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model_chk(img_torch)
    model_chk.train()
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)





model_chk = load_checkpoint(checkpoint)  
print(model_chk)

probs,classes = predict(image_path,model_chk,topk=topk_val)
class_dict = model_chk.class_to_idx
inverted_dict = dict([[v,k] for k,v in class_dict.items()])
cllist = []
for cl in classes[0].tolist():
    cllist.append(inverted_dict.get(cl))
    
cllabel=[]
cat_to_name={}
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    for cll in cllist:
        cllabel.append(cat_to_name.get(cll))
        
print("Here are top "+str(topk_val)+" Predictions \n")
for clf,prf in zip(cllabel,probs[0].tolist()):
    print(clf + " : "+ str(prf))

print("Program Execution completed")
    
