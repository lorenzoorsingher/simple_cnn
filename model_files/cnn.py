from collections import OrderedDict
import torch.nn as nn

"""
all 3 methods SHOULD be equivalent since this is a fully sequential CNN
"""

def get_training_model():

    cnnModel = nn.Sequential(OrderedDict([
        
        ("input_conv", nn.Conv2d(1,16,kernel_size=(5,5),stride=1,padding=2)), #([(W - K + 2P)/S] + 1) [14,14]??
        ("relu_1", nn.ReLU()),
        ("maxpool_1", nn.MaxPool2d(2)), #9408
        ("conv_2", nn.Conv2d(16,32,kernel_size=(5,5),stride=1,padding=2)),
        ("relu_2", nn.ReLU()),
        ("maxpool_2", nn.MaxPool2d(2)), 
        ("flatten", nn.Flatten()),
        ("linear_1", nn.Linear(32 * 7 * 7, 10))
    ]))
    return cnnModel


class CLSF2(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
            
        
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
            nn.Flatten(),
            # fully connected layer, output 10 classes
            nn.Linear(32 * 7 * 7, 10)
        )


class CLSF(nn.Module):
    def __init__(self):
        super(CLSF, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)    
        output = self.out(x)
        return output, x    # return x for visualization
