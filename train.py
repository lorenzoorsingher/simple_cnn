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
N_CLASSES = 10

transform = transforms.Compose([ transforms.ToTensor()])
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True,transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
data_loader = DataLoader(dataset=mnist_trainset,batch_size=BATCH_SIZE,shuffle=True)

opt = SGD(cnn.parameters(), lr=LR)
lossFunc = nn.NLLLoss()


def get_info(predictions, label):
	
    mat = np.zeros((N_CLASSES,N_CLASSES), np.uint8)

    for i in range(BATCH_SIZE):
        mat[torch.argmax(predictions[i]).item()][label[i]] += 1

    for line in mat:
        print('\t'.join(map(str, line)))
        print("")


    for i in range(N_CLASSES):
        print("[ANALYZING] Class: ", i)
        TP = mat[i][i]
        TN = 0
        FP = 0
        FN = 0

        for j in [x for x in range(N_CLASSES) if x != i]:
            for z in [x for x in range(N_CLASSES) if x != i]:
                TN += mat[j][z]
        
        for j in [x for x in range(N_CLASSES) if x != i]:
            FP += mat[j][i]
        
        for j in [x for x in range(N_CLASSES) if x != i]:
            FP += mat[i][j]

        prec = TP/(TP+FP)
        rec = TP/(TP+FN)
        acc = (TP+TN)/(TP+TN+FP+FN)
        f1 = (2 * prec*rec)/(prec+rec)
        print("TP: ", TP,", TN: ", TN,", FP: ", FP,", FN: ", FN)
        print("prec:\t", round(prec,2))
        print("rec:\t", round(rec,2))
        print("acc:\t", round(acc,2))
        print("F1:\t", round(f1,2))




    breakpoint()
    # return precision, recall, accuracy
    return 


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
            get_info(predictions,label)
            #breakpoint()
            stop = False
        epochLoss += loss.item()
        batchItems += len(data)
    
    print("loss: ", epochLoss/batchItems)