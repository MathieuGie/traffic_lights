import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import copy


#DROPOUT
drop=0.2

class NN_demand(nn.Module):

    def __init__(self):
        super(NN_demand, self).__init__()

        self.network=nn.Sequential(
            nn.Linear(58, 256),
            nn.LeakyReLU(),
            nn.Dropout(drop),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(drop),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(drop),
            nn.Linear(64, 8),
            nn.LeakyReLU()
        )

    def forward(self,conc_demand):
        return self.network(conc_demand)


class NN_final(nn.Module):

    def __init__(self):
        super(NN_final, self).__init__()

        self.network=nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(32, 8),
            nn.LeakyReLU()
        )

    def forward(self,conc_demand):
        return self.network(conc_demand)

class FRAP5(nn.Module):

    def __init__(self):
        
        super(FRAP5, self).__init__()

        self.NN_demand=NN_demand()
        self.NN_final=NN_final()

    def forward(self, waiting:torch.Tensor, phase:torch.Tensor, wtime_mu:torch.Tensor, wtime_sigma:torch.Tensor, wtime_max:torch.Tensor, i_position:torch.Tensor, j_position:torch.Tensor):

        #print("HELLOOOOOO")
        phase=phase
        waiting=waiting
        wtime_max=wtime_max
        wtime_mu=wtime_mu
        wtime_sigma=wtime_sigma
        i_position=i_position
        j_position=j_position
        #print(torch.cat((i_position[:,:], j_position[:,:], wtime_mu[:,:], wtime_sigma[:,:], wtime_max[:,:], waiting[:,:],phase),1))

        demand=self.NN_demand(torch.cat((i_position[:,:], j_position[:,:], wtime_mu[:,:], wtime_sigma[:,:], wtime_max[:,:], waiting[:,:],phase),1))
        #print("demand",demand)
        #return self.NN_final(demand)
        return demand