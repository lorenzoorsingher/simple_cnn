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
breakpoint()
print(cnn)

BATCH_SIZE = 64
LR = 1e-2

transform = transforms.Compose([ transforms.ToTensor()])
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True,transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
data_loader = DataLoader(dataset=mnist_trainset,batch_size=BATCH_SIZE,shuffle=True)

opt = SGD(cnn.parameters(), lr=LR)
lossFunc = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=torch.tensor([11]))

cnn.train()
for batch_id, (data,label) in enumerate(data_loader):

    (data,label) = (data.to(DEVICE), label.to(DEVICE))
    
    predictions = cnn(data)
    breakpoint()
    #print(predictions)
    loss = lossFunc(predictions, label)
    
    # zero the gradients accumulated from the previous steps,
    # perform backpropagation, and update model parameters
    opt.zero_grad()
    loss.backward()
    opt.step()