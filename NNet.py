import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class NNet(nn.Module):

    def __init__(self, n_in, n_out, hlayers=(128, 256, 128), act_fn=F.relu):
        super(NNet, self).__init__()
        self.n_hlayers = len(hlayers)
        self.fcs = nn.ModuleList([nn.Linear(n_in, hlayers[i]) if i == 0 else
                                  nn.Linear(hlayers[i-1], n_out) if i == self.n_hlayers else
                                  nn.Linear(hlayers[i-1], hlayers[i]) for i in range(self.n_hlayers+1)])
        self.act_fn = act_fn

    def forward(self, x):
        x = x.contiguous().view(-1, np.prod(x.size()[1:])) # all dimensions except the batch dimension                                                                                                    
        for i in range(self.n_hlayers):
            x = self.act_fn(self.fcs[i](x))
        x = self.fcs[-1](x)
        return x
