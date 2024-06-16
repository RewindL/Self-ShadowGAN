import torch.nn as nn
import torch.nn.functional as F
import functools
import pdb

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=2),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=2),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)
        

    def forward(self, x):
        x =  self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1) #global avg pool

class HistDiscriminator(nn.Module):
    def __init__(self, input_features):
        super(HistDiscriminator, self).__init__()
        self.mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_features, input_features // 2),
                nn.Dropout(p=0.5),
                nn.ReLU(),
                nn.Linear(input_features // 2, input_features // 4),
                nn.Dropout(p=0.5),
                nn.ReLU(),
                nn.Linear(input_features // 4, input_features // 16),
                nn.Dropout(p=0.5),
                nn.ReLU(),
                nn.Linear(input_features // 16, input_features // 64),
                nn.Dropout(p=0.5),
                nn.ReLU(),
                nn.Linear(input_features // 64,  1),
            )
        
    def forward(self, x):
        y = self.mlp(x)
        return y

class HistDiscriminator_CNN(nn.Module):
    def __init__(self, chn, bins, mode='1'):
        super(HistDiscriminator_CNN, self).__init__()
        if(mode == '1'): in_feature = 1984
        if(mode == '2'): in_feature = 2304
        self.model = nn.Sequential(
                nn.Conv1d(chn, 16, 3, 2),
                nn.ReLU(),
                nn.Conv1d(16, 32, 3, 2),
                nn.ReLU(),
                nn.Conv1d(32, 64, 3, 2),
                nn.Flatten(),
                nn.Linear(in_feature, 160),
                nn.Dropout(p=0.5),
                nn.ReLU(),
                nn.Linear(160, 32),
                nn.Dropout(p=0.5),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
        
    def forward(self, x):
        y = self.model(x)
        return y