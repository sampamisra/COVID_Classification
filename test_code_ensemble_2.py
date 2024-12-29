

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
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
torch.cuda.empty_cache()
import pandas as pd
from torch.optim.lr_scheduler import StepLR
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
data_dir = "./Test_Dataset"
test_dir=data_dir + '/Refined_Test'

# Number of classes in the dataset
num_classes = 3

# Batch size for training (change depending on how much memory you have)
batch_size = 16

feature_extract = True
sm = nn.Softmax(dim = 1)
if __name__ == '__main__':
    def test(model, criterion):
        running_corrects = 0
        running_loss=0
            #test_loss = 0.
        pred = []
        true = []
        #output =[]
        pred_wrong = []
        true_wrong = []
        image = []
        for i, (inputs, labels) in enumerate(test_loader):
           # inputs = inputs.to(device)
            #labels = labels.to(device)
            model.eval()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
                #test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
            
            #outputs = sm(outputs)
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
                #output.append(outputs[i])
                if(preds[i]!=labels[i]):
                    pred_wrong.append(preds[i])
                    true_wrong.append(labels[i])
                    image.append(inputs[i])
            
    
        # out=[]
        # for k in range(len(true)):
        #   abc=output[k].cpu()
        #   xyz=abc.detach().squeeze(-1).numpy()
        #   out.append(xyz)
        # out=np.asarray(out)        
        #mat_confusion=confusion_matrix(true, pred)
            #f1_score = f1_score(true,pred)
        # print('Confusion Matrix:\n',mat_confusion)
        #     #print('Precision: {},Recall: {}, Accuracy: {}'.format(precision*100,recall*100,accuracy*100))
        # acc=accuracy_score(true, pred)
        # print( 'Accuracy: %.3f' % acc )
        # score_precision=precision_score(true, pred)
        # print( 'Precision or PPV: %.3f' % score_precision)
        # score_recall=recall_score(true, pred)
        # print( 'Sensitivity or Recall: %.3f' % score_recall)
        # score_f1 =f1_score(true, pred)
        # print( 'F1: %.3f' % score_f1 ) 
        matrix = confusion_matrix(true, pred)
        acc=accuracy_score(true, pred)
        print( 'Accuracy: %.3f' % acc )
        matrix = matrix.astype('float')
#cm_norm = matrix / matrix.sum(axis=1)[:, np.newaxis]
        print(matrix)
#class_acc = np.array(cm_norm.diagonal())
        # class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
        # print('Sens Normal: {0:.3f}, Pneumonia: {1:.3f}, COVID-19: {2:.3f}'.format(class_acc[0],
        #                                                                    class_acc[1],
        #                                                                    class_acc[2]))
                                                                           
        # ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
        # print('PPV Normal: {0:.3f}, Pneumonia {1:.3f}, COVID-19: {2:.3f}'.format(ppvs[0],
        #                                                                  ppvs[1],
        #                                                                  ppvs[2]))
        target_names = ['class Covid','Normal', 'Pneumonia']
        print(classification_report(true, pred, target_names=target_names))
        kappa = cohen_kappa_score(true, pred)
        print('Cohens kappa: %f' % kappa)
        mc= matthews_corrcoef(true, pred)
        print('Correlation coeff: %f' % mc)

        return

        
        #print('Precision: {}, Recall: {} '.format(precision*100, recall*100))
     

    def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = True    
    
    #print(model)                        
num_epochs =200
class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, nb_classes=3):
        super(MyEnsemble, self).__init__()
        self.modelA= modelA
        self.modelB = modelB
        # Remove last linear layer
        self.modelA.fc = nn.Identity()
        self.modelB.fc =nn.Identity()
        # Create new classifier
        self.classifier = nn.Linear(1024, nb_classes)
        
    def forward(self, x):
        x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)        
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


modelA.load_state_dict(torch.load('Model\checkpoint_D1.pt'))
modelB.load_state_dict(torch.load('Model\checkpoint_D2.pt'))
#modelC.load_state_dict(torch.load('Model\checkpoint_D3.pt'))

model = MyEnsemble(modelA, modelB)
model.load_state_dict(torch.load('Model\checkpoint_ensemble12.pt'))

test_transforms = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], 
                                         [0.229, 0.224, 0.225])
                    #transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
                    ])


#targets = datasets.ImageFolder.targets

test_data= datasets.ImageFolder(test_dir,transform=test_transforms)

#targets = datasets.ImageFolder.targets

num_workers = 0

print("Number of Samples in Test ",len(test_data))

test_loader = torch.utils.data.DataLoader(test_data, batch_size, 
     num_workers=num_workers, shuffle=False)


# Setup the loss fxn
criterion = nn.CrossEntropyLoss()        
torch.cuda.empty_cache()
test(model, criterion) 

