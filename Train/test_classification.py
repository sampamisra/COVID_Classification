# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:27:45 2021

@author: Sampa
"""
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
import csv
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef
torch.cuda.empty_cache()
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as pl
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
data_dir = "./DatasetD"
test_dir=data_dir + '/Test'
num_classes = 3
batch_size =32  
model = "Resnet18_1"
if __name__ == '__main__':  
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


            else:
                print("Invalid model name, exiting...")
                exit()

            return model    
    def test(model, criterion):
        model.to(device) 
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
        out=[]
        for k in range(len(true)):
          abc=output[k].cpu()
          xyz=abc.detach().squeeze(-1).numpy()
          out.append(xyz)
        out=np.asarray(out)        
        mat_confusion=confusion_matrix(true, pred)
            #f1_score = f1_score(true,pred)
        print('Confusion Matrix:\n',mat_confusion)
            #print('Precision: {},Recall: {}, Accuracy: {}'.format(precision*100,recall*100,accuracy*100))
        score_precision = mat_confusion[1,1]/( mat_confusion[1,1] + mat_confusion[0,1] )*100
        acc = (mat_confusion[1,1]+mat_confusion[0,0])/( mat_confusion[1,1] + mat_confusion[0,1] + mat_confusion[1,0] + mat_confusion[0,0] )*100


        # REC = TP/(TP+FN)
        score_recall    = mat_confusion[1,1]/( mat_confusion[1,1] + mat_confusion[1,0] )*100
        # specificity = TN/(TN+FP)
        specificity   = mat_confusion[0,0]/( mat_confusion[0,0] + mat_confusion[0,1] )*100
        #NPV  = TN/(TN+FN)
        NPV = mat_confusion[0,0]/( mat_confusion[0,0] + mat_confusion[1,0] )*100
            
            # F1 = 2*PRE*( REC/(PRE+REC)
        score_f1 = 2*score_precision*( score_recall/(score_precision+score_recall) )
        print( 'Accuracy: %.3f' % acc )
        print( 'Precision or PPV: %.3f' % score_precision )
        print( 'NPV: %.3f' % NPV )
        print( 'Specificity %.3f' % specificity)
        print( 'Sensitivity or Recall: %.3f' % score_recall )
        print( 'F1: %.3f' % score_f1 ) 
        kappa = cohen_kappa_score(true, pred)
        print('Cohens kappa: %f' % kappa)
        mc= matthews_corrcoef(true, pred)
        print('Correlation coeff: %f' % mc)
        # ROC AUC
    
        # confusion matrix
       
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

        return out 
model = initialize_model(model, num_classes)
model.load_state_dict(torch.load('model\checkpoint_D4_0.pt'))
feature_extract = True
sm = nn.Softmax(dim = 1)
test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                         [0.229, 0.224, 0.225])
                    #transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
                    ])
    
                    
test_data= datasets.ImageFolder(test_dir,transform=test_transforms)
num_workers = 0
print("Number of Samples in Test ",len(test_data))
test_loader = torch.utils.data.DataLoader(test_data, batch_size, 
     num_workers=num_workers, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#optimizer= optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
criterion = nn.CrossEntropyLoss()
out=test(model, criterion) 
# images_path -> [ [images path, label] * 835 ]

# with open(f"majority.csv", "w") as f:
#     wr = csv.writer(f)
#     wr.writerow(["file", "prob_0", "prob_1", "prob_2", "prob_3", "prob_4",  "pred", "label"])
#     for i in range(len(preds)):
#         f = os.path.basename(images_path[i][0])
#         prob_0 = round(soft[i][0], 6)
#         prob_1 = round(soft[i][1], 6)
#         prob_2 = round(soft[i][2], 6)
#         prob_3 = round(soft[i][3], 6)
#         prob_4 = round(soft[i][4], 6)
#         pred = preds[i]
#         label = true[i]
#         wr.writerow([f, prob_0, prob_1, prob_2, prob_3, prob_4, pred, label])

