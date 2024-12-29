from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from torch.utils import data
import os
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
from early import EarlyStopping

data_dir = "./DatasetD"
test_dir=data_dir + '/test'

num_classes = 3

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Number of epochs to train for
num_epochs =200
class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, modelC, nb_classes=3):
        super(MyEnsemble, self).__init__()
        self.modelA= modelA
        self.modelB = modelB
        self.modelC = modelC
        # Remove last linear layer
        self.modelA.fc = nn.Identity()
        self.modelB.fc =nn.Identity()
        self.modelC.fc =nn.Identity()
        # Create new classifier
        self.classifier = nn.Linear(1536, nb_classes)
        
    def forward(self, x):
        x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        x3 = self.modelC(x)
        x3 = x3.view(x3.size(0), -1)
        x = torch.cat((x1, x2, x3), dim=1)
        
        x = self.classifier(F.relu(x))
        return x

# Train your separate models
# ...
# We use pretrained torchvision models here
modelA = models.resnet18(pretrained=True)

            #weight = model.conv1.weight.clone()
num_ftrs = modelA.fc.in_features
# modelA.fc = nn.Sequential(
#                 nn.Dropout(0.5),
#                 nn.Linear(num_ftrs, 2))
                
modelA.fc = nn.Linear(num_ftrs, 2)
modelB = models.resnet18(pretrained=True)

            #weight = model.conv1.weight.clone()
num_ftrs = modelB.fc.in_features
# modelB.fc = nn.Sequential(
#                 nn.Dropout(0.5),
#                 nn.Linear(num_ftrs, 2))

# input_size = 224

modelB.fc = nn.Linear(num_ftrs, 2)
modelC = models.resnet18(pretrained=True)

            #weight = model.conv1.weight.clone()
num_ftrs = modelC.fc.in_features
# modelC.fc = nn.Sequential(
#                 nn.Dropout(0.5),
#                 nn.Linear(num_ftrs, 2))

modelC.fc = nn.Linear(num_ftrs, 2)

modelA.load_state_dict(torch.load('checkpoint_D1.pt'))
modelB.load_state_dict(torch.load('checkpoint_D2.pt'))
modelC.load_state_dict(torch.load('checkpoint_D3.pt'))

model = MyEnsemble(modelA, modelB, modelC)
feature_extract = True
sm = nn.Softmax(dim = 1)
from early import EarlyStopping

if __name__ == '__main__':
    def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, patience):
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
                    logps = model.forward(inputs)
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
        model.load_state_dict(torch.load('checkpoint.pt'))
        return  model, avg_train_losses, avg_valid_losses,  train_acc, valid_acc
    
   
    def test(model, criterion):
        running_corrects = 0
        running_loss=0
            #test_loss = 0.
        pred = []
        true = []
        output =[]
        pred_wrong = []
        true_wrong = []
        image = []
        for j, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.eval()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
                #test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
            
            outputs = sm(outputs)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            preds = np.reshape(preds,(len(preds),1))
            labels = np.reshape(labels,(len(preds),1))
            inputs = inputs.cpu().numpy()
                
            for i in range(len(preds)):
                pred.append(preds[i])
                true.append(labels[i])
                output.append(outputs[i])
                if(preds[i]!=labels[i]):
                    pred_wrong.append(preds[i])
                    true_wrong.append(labels[i])
                    image.append(inputs[i])
            
            # epoch_acc = running_corrects.double()/(len(test_loader)*batch_size)
            # epoch_loss = running_loss/(len(test_loader)*batch_size)
            # print('Accuracy, loss'.format(epoch_acc,epoch_loss))
            # precision = precision_score(true,pred)
            # recall = recall_score(true,pred)
            # accuracy = accuracy_score(true,pred)
        # out=[]
        # for k in range(len(true)):
        #   abc=output[k].cpu()
        #   xyz=abc.detach().squeeze(-1).numpy()
        #   out.append(xyz)
        # out=np.asarray(out)        
        mat_confusion=confusion_matrix(true, pred)
            #f1_score = f1_score(true,pred)
        print('Confusion Matrix:\n',mat_confusion)
            #print('Precision: {},Recall: {}, Accuracy: {}'.format(precision*100,recall*100,accuracy*100))
        acc=accuracy_score(true, pred)
        print( 'Accuracy: %.3f' % acc )
        
        # kappa = cohen_kappa_score(true, pred)
        # print('Cohens kappa: %f' % kappa)
        # mc= matthews_corrcoef(true, pred)
        # print('Correlation coeff: %f' % mc)
        # # ROC AUC
        # auc = roc_auc_score(np.asarray(true).ravel(), out[:,1])
        # print('ROC AUC: %f' % auc)
        # # confusion matrix
       
        # fpr, tpr, thresholds = roc_curve(np.asarray(true).ravel(), out[:,1])
    
        # plt.figure()
        # plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc)
        # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        # #plt.xlim([0.0, 1.0])
        # #plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic')
        # plt.legend(loc="lower right")
        # plt.show()

        return 

        
        #print('Precision: {}, Recall: {} '.format(precision*100, recall*100))

                          


train_transforms = transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
                                      transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(15),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                      ])


test_transforms = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], 
                                         [0.229, 0.224, 0.225])
                    #transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
                    ])


validation_transforms = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])
merge_data = datasets.ImageFolder(data_dir + "/train", transform=train_transforms)

train_data, valid_data = train_test_split(merge_data, test_size = 0.2, random_state= 123)
test_data= datasets.ImageFolder(test_dir,transform=test_transforms)



#targets = datasets.ImageFolder.targets

num_workers = 0
# percentage of training set to use as validation
# valid_size = 0.2

# test_size = 0.2

# # obtain training indices that will be used for validation
# num_train = len(master)
# indices = list(range(num_train))
# np.random.shuffle(indices)
# valid_split = int(np.floor((valid_size) * num_train))
# test_split = int(np.floor((valid_size+test_size) * num_train))
# valid_idx, test_idx, train_idx = indices[:valid_split], indices[valid_split:test_split], indices[test_split:]

# #print(len(valid_idx), len(test_idx), len(train_idx))
#print("Total Number of Samples: ",len(train))
print("Number of Samples in Train: ",len(train_data))
print("Number of Samples in Valid: ",len(valid_data))
print("Number of Samples in Test ",len(test_data))
#print("Number of Classes: ",len(master.classes))

# define samplers for obtaining training and validation batches


# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size,
     num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size, 
     num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size, 
     num_workers=num_workers, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params_to_update = model.parameters()
#print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            #print("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
patience = 10
optimizer= optim.SGD(params_to_update, lr=0.001, momentum=0.8)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)

# Setup the loss fxn
weight = torch.tensor([0.11, 0.04, 1])
class_weights = torch.FloatTensor(weight).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights) 
#criterion = nn.CrossEntropyLoss()
model, train_loss, valid_loss, train_acc, valid_acc=train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, patience)
# fig = plt.figure(figsize=(10,8))
# plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
# plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

# # find position of lowest validation loss
# minposs = valid_loss.index(min(valid_loss))+1 
# plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

# plt.xlabel('Epochs..........>')
# plt.ylabel('Loss..........>')
# plt.xlim(0, len(train_loss)+1) # consistent scale
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
# fig.savefig('loss_plotdataset2.png', bbox_inches='tight')

# fig = plt.figure(figsize=(10,8))
# plt.plot(range(1,len(train_acc)+1),train_acc, label='Training Accuracy')
# plt.plot(range(1,len(valid_acc)+1),valid_acc,label='Validation Accuracy')
# minposs = valid_loss.index(min(valid_loss))+1 
# plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

# #find position of lowest validation loss

# plt.xlabel('Epochs..........>')
# plt.ylabel('Accuracy..........>')
# plt.xlim(0, len(train_acc)+1) # consistent scale
# #plt.grid(True)
# plt.legend()
# #plt.tight_layout()
# plt.show()
# fig.savefig('accplotdataset2.png', bbox_inches='tight')
# #test(model, criterion) 

