import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(0)

batchSizeTrain = 100
batchSizeTest = 1000
numberOfCommunicationsRound = 1000
learningRate = 1e-4

transform = transforms.Compose(
    [transforms.ToTensor()])

trainDataset = torchvision.datasets.MNIST(root='data', 
                                    train=True, 
                                    transform=transform,  
                                    download=True)
testDataset = torchvision.datasets.MNIST(root='data', 
                                    train=False, 
                                    transform=transform) 

testLoader = torch.utils.data.DataLoader(dataset=testDataset, 
                                        batch_size=batchSizeTest, 
                                        shuffle=False)

class neural_network(nn.Module):
    def __init__(self):
        super(neural_network, self).__init__()
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

model = neural_network()
if torch.cuda.is_available():
    model.cuda()

imsAll = trainDataset.train_data
lblsAll = trainDataset.train_labels
imsAll = imsAll[:, None, :, :]
imsAll = imsAll.float()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_fn = nn.CrossEntropyLoss()
for x in range(numberOfCommunicationsRound):
    indices = torch.randint(0, 60000, (batchSizeTrain,))
    imgs = imsAll[indices,:,:,:]
    lbls = lblsAll[indices]

    imgs, lbls = imgs.to(device), lbls.to(device)

    outp = model(imgs)
    loss = loss_fn(outp, lbls)

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param.data -= learningRate * torch.sign(param.grad)

    print ('Round: ' + str(x+1)+ f', Loss: {loss.item():.4f}')


    if (x) % 50 == 51:
        acc = torch.Tensor(int(10000/batchSizeTest))
        for indTestBatch, (imgsTest, lblsTest) in enumerate(testLoader):
            imgsTest, lblsTest = imgsTest.to(device), lblsTest.to(device)
                
            outp = model(imgsTest)
            _, preds = torch.max(outp, dim = 1)
            acc[indTestBatch] = (torch.tensor(torch.sum(preds == lblsTest).item()/ len(preds)))
        print ('Round: ' + str(x+1)+ f', Acc: {torch.mean(acc).item():.4f}')

print('done')