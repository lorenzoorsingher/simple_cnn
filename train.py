# import the necessary packages
from model_files import cnn
from model_files.cnn import CLSF, CLSF2
import torch
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import SGD
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}...".format(DEVICE))


#breakpoint()
# initialize our model and display its architecture
#cnn = cnn.get_training_model().to(DEVICE)
cnn = CLSF2().to(DEVICE)

model_parameters = filter(lambda p: p.requires_grad, cnn.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params, " total params")
print(cnn)

EPOCH = 100
BATCH_SIZE = 64
LR = 1e-2

transform = transforms.Compose([ transforms.ToTensor()])
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True,transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
data_loader = DataLoader(dataset=mnist_trainset,batch_size=BATCH_SIZE,shuffle=True)

opt = SGD(cnn.parameters(), lr=LR)
lossFunc = nn.NLLLoss()





cnn.train()
for i in range(EPOCH):
    print("EPOCH n",i,"\n")

    epochLoss = 0
    batchItems = 0
    stop = True
    for batch_id, (data,label) in enumerate(data_loader):

        (data,label) = (data.to(DEVICE), label.to(DEVICE))
        
        predictions = cnn(data)

        loss = lossFunc(predictions, label)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        if stop:
            breakpoint()
            stop = False
        epochLoss += loss.item()
        batchItems += len(data)
    
    print("loss: ", epochLoss/batchItems)