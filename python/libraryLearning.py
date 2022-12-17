import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import scipy.io

class objNeuralNetwork(nn.Module):
    def __init__(self):
        super(objNeuralNetwork, self).__init__()
        torch.manual_seed(0)
        self.conv5l = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=20,            
                kernel_size=5,              
                stride=(1, 1),                   
                padding=(0, 0),                  
            ),                              
            nn.BatchNorm2d(20),                   
            nn.ReLU(),                      
        )

        self.conv3l = nn.Sequential(         
            nn.Conv2d(
                in_channels=20,              
                out_channels=20,            
                kernel_size=3,              
                stride=(1, 1),                   
                padding=(1, 1),                  
            ),                              
            nn.BatchNorm2d(20),                   
            nn.ReLU(),                      
        )

        self.conv3l_2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=20,              
                out_channels=20,            
                kernel_size=3,              
                stride=(1, 1),                   
                padding=(1, 1),                  
            ),                              
            nn.BatchNorm2d(20),                   
            nn.ReLU(),                      
        )

        self.out = nn.Sequential(         
            nn.Linear(11520, 10)                              
        )

    def forward(self, y):
        y = self.conv5l(y)
        y = self.conv3l(y)
        y = self.conv3l_2(y)
        y = y.view(y.size(0), -1)       
        outp = self.out(y)
        return outp  


    
class objNeuralNetwork2(nn.Module):
    
    def __init__(self):
        super(objNeuralNetwork2, self).__init__()
        torch.manual_seed(0)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 10, kernel_size=5),nn.BatchNorm2d(10))
        self.conv2 = nn.Sequential( nn.Conv2d(10, 20, kernel_size=5),nn.BatchNorm2d(20))
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x  

class objNeuralNetwork3(nn.Module):
    def __init__(self):
        super(objNeuralNetwork3, self).__init__()
        torch.manual_seed(0)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class objNeuralNetwork4(nn.Module):
    def __init__(self):
        super(objNeuralNetwork4, self).__init__()
        torch.manual_seed(0)
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),         
            nn.BatchNorm2d(16),  
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.BatchNorm2d(32),  
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output    # return x for visualization


def fcn_dataLoader(parameters):
    targets = parameters['labels']
    indexShard = parameters['indexShard']
    numberOfImagesPerLabel = parameters['numberOfImagesPerLabel']

    transform = transforms.Compose([transforms.ToTensor()])

    trainDataset = torchvision.datasets.MNIST(root='data', 
                                        train=True, 
                                        transform=transform,  
                                        download=True)
    imagesAll = torch.Tensor(numberOfImagesPerLabel*len(targets),28,28)
    labelsAll = torch.Tensor(numberOfImagesPerLabel*len(targets)).long()
    for indTarget in np.arange(len(targets)):
        indices = trainDataset.targets == targets[indTarget] # if you want to keep images with the label 5
        imgs, lbls = trainDataset.data[indices], trainDataset.targets[indices]
        indicesImgs = np.arange(numberOfImagesPerLabel) + indexShard*numberOfImagesPerLabel
        indices = np.arange(numberOfImagesPerLabel) + indTarget*numberOfImagesPerLabel
        imagesAll[indices,:,:] = imgs[indicesImgs,:,:].float()
        labelsAll[indices] = lbls[indicesImgs]

    return imagesAll[:, None, :, :], labelsAll


def fcn_calculateTestAccuracy(model,device):
    batchSizeTest = 250

    transform = transforms.Compose([transforms.ToTensor()])

    testDataset = torchvision.datasets.MNIST(root='data', 
                                        train=False, 
                                        transform=transform) 

    testLoader = torch.utils.data.DataLoader(dataset=testDataset, 
                                            batch_size=batchSizeTest, 
                                            shuffle=False)

    acc = torch.Tensor(int(10000/batchSizeTest))
    for indTestBatch, (imgsTest, lblsTest) in enumerate(testLoader):
        imgsTest, lblsTest = imgsTest.to(device), lblsTest.to(device)
                
        outp = model(imgsTest)
        _, preds = torch.max(outp, dim = 1)
        acc[indTestBatch] = (torch.tensor(torch.sum(preds == lblsTest).item()/ len(preds)))
    return torch.mean(acc).item()


class objLearningEngine():
    def __init__(self, parameters):

        self.model = objNeuralNetwork4()
        if torch.cuda.is_available():
            self.model.cuda()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.batchSizeTrain = parameters['batchSizeTrain']
        self.learningRate = parameters['learningRate']
        self.imagesTraining = parameters['imagesTraining']
        self.labelsTraining = parameters['labelsTraining']
        self.signatureForSaving = parameters['signatureForSaving']
        
        self.loss_fn = nn.CrossEntropyLoss()

        self.lossChange = []
        self.accChange = []

        self.numberOfParameters = 0
        self.numberOfTrainingImages = torch.numel(self.labelsTraining)
        with torch.no_grad():
            for param in self.model.parameters():
                self.numberOfParameters += torch.numel(param)
        print('# of parameters:'+ str(self.numberOfParameters))

    def step(self, gradientsInput):
        Ncurrent = 0 
        with torch.no_grad():
            for param in self.model.parameters():
                numberOfParametersLayer = torch.numel(param)
                param.data -= self.learningRate * (torch.from_numpy(gradientsInput[Ncurrent + np.arange(numberOfParametersLayer)]).view(param.data.shape)).to(self.device)
                Ncurrent += numberOfParametersLayer

        indices = torch.randint(0, self.numberOfTrainingImages, (self.batchSizeTrain,))
        imgs = self.imagesTraining[indices,:,:,:]
        lbls = self.labelsTraining[indices]
        imgs, lbls = imgs.to(self.device), lbls.to(self.device)

        outp = self.model(imgs)
        loss = self.loss_fn(outp, lbls)

        self.lossChange.append(loss.item())
        rateForAcc = 5
        if len(self.lossChange) % rateForAcc == 1:
            acc = fcn_calculateTestAccuracy(self.model,self.device)
            self.accChange.append(acc)
            print('Update round: ' + str(len(self.lossChange)) + ', Acc:' + str(acc))

            

        print('Update round: ' + str(len(self.lossChange)) + ', Loss:' + str(loss.item()))

        recording = True
        if recording == True:
            scipy.io.savemat(self.signatureForSaving + '_lossAndAcc' + '.mat', mdict={'lossChange': self.lossChange, 'accChange': self.accChange, 'rateForAcc': rateForAcc})


        self.model.zero_grad()
        loss.backward()

        gradientsOutput = torch.Tensor(self.numberOfParameters)
        Ncurrent = 0 
        with torch.no_grad():
            for param in self.model.parameters():
                numberOfParametersLayer = torch.numel(param)
                gradientsLayer = param.grad
                gradientsOutput[Ncurrent + torch.arange(numberOfParametersLayer)] = gradientsLayer.view(-1).to('cpu') 
                Ncurrent += numberOfParametersLayer
        return gradientsOutput.numpy()
