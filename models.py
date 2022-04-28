import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from GraphFunctions import *

# Original authors are: (some parts are modified)
__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"


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


class MPNN(nn.Module):
    """
        MPNN as proposed by Gilmer et al..

        This class implements the whole Gilmer et al. model following the functions Message, Update and Readout.

        Parameters
        ----------
        in_n : int list
            Sizes for the node and edge features.
        hidden_state_size : int
            Size of the hidden states (the input will be padded with 0's to this size).
        message_size : int
            Message function output vector size.
        n_layers : int
            Number of iterations Message+Update (weight tying).
        l_target : int
            Size of the output.
        type : str (Optional)
            Classification | [Regression (default)]. If classification, LogSoftmax layer is applied to the output vector.
    """

    def __init__(self, in_n, hidden_state_size, message_size, n_layers, l_target, type='regression'):
        super(MPNN, self).__init__()

        # Define message
        self.m = nn.ModuleList(
            [MessageFunction('mpnn', args={'edge_feat': in_n[1], 'in': hidden_state_size, 'out': message_size})])

        # Define Update
        self.u = nn.ModuleList([UpdateFunction('mpnn',
                                               args={'in_m': message_size,
                                                     'out': hidden_state_size})])

        # Define Readout
        self.r = ReadoutFunction('mpnn',
                                 args={'in': hidden_state_size,
                                       'target': l_target})

        self.type = type

        self.args = {}
        self.args['out'] = hidden_state_size

        self.n_layers = n_layers

    def forward(self, g, h_in, e):

        h = []

        # Padding to some larger dimension d
        h_t = torch.cat([h_in, Variable(
            torch.zeros(h_in.size(0), h_in.size(1), self.args['out'] - h_in.size(2)).type_as(h_in.data))], 2)

        h.append(h_t.clone())

        # Layer
        for t in range(0, self.n_layers):
            e_aux = e.view(-1, e.size(3))

            h_aux = h[t].view(-1, h[t].size(2))

            m = self.m[0].forward(h[t], h_aux, e_aux)
            m = m.view(h[0].size(0), h[0].size(1), -1, m.size(1))

            # Nodes without edge set message to 0
            m = torch.unsqueeze(g, 3).expand_as(m) * m

            m = torch.squeeze(torch.sum(m, 1))

            h_t = self.u[0].forward(h[t], m)

            # Delete virtual nodes
            h_t = (torch.sum(h_in, 2)[...,None].expand_as(h_t) > 0).type_as(h_t) * h_t
            h.append(h_t)

        # Readout
        res = self.r.forward(h)

        if self.type == 'classification':
            res = nn.LogSoftmax()(res)
        return res
