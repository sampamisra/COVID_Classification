from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.optim import optimizer
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from torch.utils import data
import os
from sklearn.model_selection import KFold
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
torch.cuda.empty_cache()
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as pl
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
data_dir = "./DatasetD"
num_classes = 3
batch_size =16
num_epochs =1000
model_name = "Resnet18_1"
feature_extract = True
learning_rate = 1e-4
from early import EarlyStopping
if __name__ == '__main__':
    def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, patience, i ):
        model.to(device) 
        epochs = num_epochs
        valid_loss_min = np.Inf
        train_losses = []
    # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = [] 
        train_acc, valid_acc =[],[]
        steps=0
        #valid_acc =[]
        best_acc = 0.0
        import time
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        for epoch in range(epochs):    
            start = time.time()
            #scheduler.step()
            model.train()
            total_train = 0
            correct_train = 0
            for inputs, labels in train_loader:
                steps+=1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)                
                optimizer.zero_grad()                
                logps = model(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                _, predicted = torch.max(logps.data, 1)
                total_train += labels.nelement()
                correct_train += predicted.eq(labels.data).sum().item()
                train_accuracy = correct_train / total_train
                model.eval()
               
            with torch.no_grad():
                accuracy = 0
                for inputs, labels in valid_loader:
                    
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model(inputs)
                    loss = criterion(logps, labels)
                    valid_losses.append(loss.item())
        # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
             
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            valid_acc.append(accuracy/len(valid_loader)) 
            train_acc.append(train_accuracy)
        
        # calculate average losses
            
            valid_accuracy = accuracy/len(valid_loader)          
            
            # print training/validation statistics 
            print(f"Epoch {epoch+1}/{epochs}.. ")
            #print('train Loss: {:.3f}'.format(epoch, loss.item()), "Training Accuracy: %d %%" % (train_accuracy))
            #print('Training Accuracy: {:.6f}'.format(
            #    train_accuracy))
            print('Training Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining Accuracy: {:.6f} \tValidation Accuracy: {:.6f}'.format(
                train_loss, valid_loss, train_accuracy*100, valid_accuracy*100))
            train_losses = []
            valid_losses = []        
            if valid_accuracy > best_acc:
                best_acc = valid_accuracy
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
           
        print('Best val Acc: {:4f}'.format(best_acc*100))  
        # model.load_state_dict(torch.load('checkpoint.pt'))
        # plt.title("Accuracy vs. Number of Training Epochs")
        # plt.xlabel("Training Epochs")
        # plt.ylabel("Accuracy")      
        # plt.plot(train_acc, label='Training acc')
        # plt.plot(valid_acc, label='Validation acc')
        # plt.legend(frameon=False)
        # plt.show()
        
        #model.load_state_dict(torch.load('checkpoint_{0}.pt'.format(i)))
        torch.save(model.state_dict(), "./model/checkpoint_D4_{0}.pt".format(i))
        print("model saved")
        
        return  avg_train_losses, avg_valid_losses,  train_acc, valid_acc
 
    def initialize_model(model_name, num_classes):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
            model = None
            input_size = 0

            if model_name == "resnet18_1":
                """ Resnet18
                """
                model = models.resnet18(pretrained=True)
                #print(model)
                #weight = model.conv1.weight.clone()
                #model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
                # with torch.no_grad():
                #     model.conv1.weight[:, :3] = weight
                #     model.conv1.weight[:, 3] = model.conv1.weight[:, 0]
                count=0
                # for child in model.children():
                #     count+=1

                # print('No. of layers')
                # print(count)
                # count = 0
                for child in model.children():
                    count+=1
                    if count <4:
                        for param in child.parameters():
                            param.requires_grad = False
                        print(count)    
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, num_classes)
                input_size = 224
                #print(model)
            elif model_name == "resnet18":
                """ Resnet18
                """
                model = models.resnet18()
                #weight = model.conv1.weight.clone()
                #model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                # with torch.no_grad():
                #     model.conv1.weight[:, :3] = weight
                #     model.conv1.weight[:, 3] = model.conv1.weight[:, 0]
                #set_parameter_requires_grad(model, feature_extract)
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, num_classes)
                input_size = 224    
                

            elif model_name == "alexnet1":
                """ Alexnet
                """
                model= models.alexnet(pretrained=True)
            # print(model)
                #weight = model.features[0].weight.clone()
                model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
                # with torch.no_grad():
                #     model.conv1.weight[:, :3] = weight
                #     model.conv1.weight[:, 3] = model.conv1.weight[:, 0]
            
                count=0
                for child in model.children():
                    count+=1
                print('No. of layers')
                print(count)
                count = 0  
                for child in model.children():
                    if count <3:
                        for param in child.parameters():
                            param.requires_grad = False  
                        count+=1        
                num_ftrs = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_ftrs,num_classes)
                input_size = 224
            elif model_name == "alexnet":
                """ Alexnet
                """
                model= models.alexnet()
                #model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
                num_ftrs = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_ftrs,num_classes)
                input_size = 224
            elif model_name == "resnet50":
                """ Resnet50
                """
                model = models.resnet50(pretrained=True)
                #set_parameter_requires_grad(model, feature_extract)
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, num_classes)
                input_size = 224
            elif model_name == "vgg1":
                """ VGG19_bn
                """
                model = models.vgg19_bn(pretrained= True)
                model.features[0] = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
                count=0
                for child in model.children():
                    count+=1
                print('No. of layers')
                print(count)
                count = 0  
                for child in model.children():
                    count+=1
                    if count <1:
                        for param in child.parameters():
                            param.requires_grad = False  
                num_ftrs = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_ftrs,num_classes)
                input_size = 224  

            elif model_name == "vgg16":
                """ VGG11_bn
                """
                model = models.vgg16(pretrained=True)
                num_ftrs = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_ftrs,2)
                layer_counter = 0
                for (name, module) in model.named_children():
                    for layer in module.children():
                        if layer_counter<20:
                            for param in layer.parameters():
                                param.requires_grad = False           
                            print('Layer "{}" in module "{}" was frozen!'.format(layer_counter, name))
                        layer_counter+=1    
                input_size = 224

            elif model_name == "squeezenet":
                """ Squeezenet
                """
                model= models.squeezenet1_0()
                model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
                model.num_classes = num_classes
                input_size = 224

            elif model_name == "densenet":
                """ Densenet
                """
                model= models.densenet169()
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, num_classes)
                input_size = 224

            elif model_name == "inception":
                """ Inception v3
                Be careful, expects (299,299) sized images and has auxiliary output
                """
                model = models.inception_v3(pretrained=True)
                #set_parameter_requires_grad(model, feature_extract)
                model.features[0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
                # Handle the auxilary net
                num_ftrs = model.AuxLogits.fc.in_features
                model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
                # Handle the primary net
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs,num_classes)
                input_size = 224

            else:
                print("Invalid model name, exiting...")
                exit()

            return model       

# Initialize the model for this run

#print(model.conv1.weight)              
sm = nn.Softmax(dim = 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patience = 200

#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)
# Setup the loss fxn
#weight = torch.tensor([0.11, 0.04, 1])
#class_weights = torch.FloatTensor(weight).cuda()
#criterion = nn.CrossEntropyLoss(weight=class_weights) 
train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(60),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                      ])

# validation_transforms = transforms.Compose([transforms.Resize((224,224)),
#                                             transforms.ToTensor(),
#                                             transforms.Normalize([0.485, 0.456, 0.406], 
#                                                                  [0.229, 0.224, 0.225])])
merge_data = datasets.ImageFolder(data_dir + "/train", transform=train_transforms)
fold_counts= 5
kfold = KFold(n_splits=fold_counts, random_state=777, shuffle=True)
num_workers = 0
#
# #--------------------------------------------------------------
for i, (train_index, validate_index) in enumerate(kfold.split(merge_data)):
    #print("train index:", train_index, "validate index:", validate_index)
    train = torch.utils.data.Subset(merge_data, train_index)
    validation = torch.utils.data.Subset(merge_data, validate_index)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    print("Number of Samples in Train: ",len(train))
    print("Number of Samples in Valid: ",len(validation))
    model = initialize_model(model_name, num_classes)
    #model.apply(reset_weights)
    optimizer= optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss() 
    train_loss, valid_loss, train_acc, valid_acc=train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, patience, i)
