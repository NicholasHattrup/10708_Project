import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from GraphFunctions import *

# Original authors are: (some parts are modified)
__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"


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


class MpnnGGNN(nn.Module):
    """
        MPNN as proposed by Li et al..

        This class implements the whole Li et al. model following the functions proposed by Gilmer et al. as
        Message, Update and Readout.

        Parameters
        ----------
        e : int list.
            Possible edge labels for the input graph.
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

    def __init__(self, e, hidden_state_size, message_size, n_layers, l_target, type='regression'):
        super(MpnnGGNN, self).__init__()

        # Define message
        self.m = nn.ModuleList([MessageFunction('ggnn', args={'e_label': e, 'in': hidden_state_size, 'out': message_size})])

        # Define Update
        self.u = nn.ModuleList([UpdateFunction('ggnn',
                                                args={'in_m': message_size,
                                                'out': hidden_state_size})])

        # Define Readout
        self.r = ReadoutFunction('ggnn',
                                 args={'in': hidden_state_size,
                                       'target': l_target})

        self.type = type

        self.args = {}
        self.args['out'] = hidden_state_size

        self.n_layers = n_layers

    def forward(self, g, h_in, e):

        h = []

        # Padding to some larger dimension d
        h_t = torch.cat([h_in, Variable(torch.Tensor(h_in.size(0), h_in.size(1), self.args['out'] - h_in.size(2)).type_as(h_in.data).zero_())], 2)

        h.append(h_t.clone())

        # Layer
        for t in range(0, self.n_layers):

            h_t = Variable(torch.zeros(h[0].size(0), h[0].size(1), h[0].size(2)).type_as(h_in.data))

            # Apply one layer pass (Message + Update)
            for v in range(0, h_in.size(1)):

                m = self.m[0].forward(h[t][:, v, :], h[t], e[:, v, :])

                # Nodes without edge set message to 0
                m = g[:, v, :, None].expand_as(m) * m

                m = torch.sum(m, 1)

                # Update
                h_t[:, v, :] = self.u[0].forward(h[t][:, v, :], m)

            # Delete virtual nodes
            h_t = (torch.sum(torch.abs(h_in), 2).expand_as(h_t) > 0).type_as(h_t)*h_t
            h.append(h_t.clone())

        # Readout
        res = self.r.forward(h)
        if self.type == 'classification':
            res = nn.LogSoftmax()(res)
        return res


class MpnnIntNet(nn.Module):
    """
        MPNN as proposed by Battaglia et al..

        This class implements the whole Battaglia et al. model following the functions proposed by Gilmer et al. as
        Message, Update and Readout.

        Parameters
        ----------
        in_n : int list
            Sizes for the node and edge features.
        out_message : int list
            Output sizes for the different Message functions.
        out_update : int list
            Output sizes for the different Update functions.
        l_target : int
            Size of the output.
        type : str (Optional)
            Classification | [Regression (default)]. If classification, LogSoftmax layer is applied to the output vector.
    """

    def __init__(self, in_n, out_message, out_update, l_target, type='regression'):
        super(MpnnIntNet, self).__init__()

        n_layers = len(out_update)

        # Define message 1 & 2
        self.m = nn.ModuleList([MessageFunction('intnet', args={'in': 2*in_n[0] + in_n[1], 'out': out_message[i]})
                                if i == 0 else
                                MessageFunction('intnet', args={'in': 2*out_update[i-1] + in_n[1], 'out': out_message[i]})
                                for i in range(n_layers)])

        # Define Update 1 & 2
        self.u = nn.ModuleList([UpdateFunction('intnet', args={'in': in_n[0]+out_message[i], 'out': out_update[i]})
                                if i == 0 else
                                UpdateFunction('intnet', args={'in': out_update[i-1]+out_message[i], 'out': out_update[i]})
                                for i in range(n_layers)])

        # Define Readout
        self.r = ReadoutFunction('intnet', args={'in': out_update[-1], 'target': l_target})

        self.type = type

    def forward(self, g, h_in, e):

        h = []
        h.append(h_in)

        # Layer
        for t in range(0, len(self.m)):

            u_args = self.u[t].get_args()
            h_t = Variable(torch.zeros(h_in.size(0), h_in.size(1), u_args['out']).type_as(h[t].data))

            # Apply one layer pass (Message + Update)
            for v in range(0, h_in.size(1)):

                m = self.m[t].forward(h[t][:, v, :], h[t], e[:, v, :, :])

                # Nodes without edge set message to 0
                m = g[:, v, :,None].expand_as(m) * m

                m = torch.sum(m, 1)

                # Interaction Net
                opt = {}
                opt['x_v'] = Variable(torch.Tensor([]).type_as(m.data))

                h_t[:, v, :] = self.u[t].forward(h[t][:, v, :], m, opt)

            h.append(h_t.clone())

        # Readout
        res = self.r.forward(h)
        if self.type == 'classification':
            res = nn.LogSoftmax()(res)
        return res


class MpnnDuvenaud(nn.Module):
    """
        MPNN as proposed by Duvenaud et al..

        This class implements the whole Duvenaud et al. model following the functions proposed by Gilmer et al. as 
        Message, Update and Readout.

        Parameters
        ----------
        d : int list.
            Possible degrees for the input graph.
        in_n : int list
            Sizes for the node and edge features.
        out_update : int list
            Output sizes for the different Update functions.
        hidden_state_readout : int
            Input size for the neural net used inside the readout function.
        l_target : int
            Size of the output.
        type : str (Optional)
            Classification | [Regression (default)]. If classification, LogSoftmax layer is applied to the output vector.
    """

    def __init__(self, d, in_n, out_update, hidden_state_readout, l_target, type='regression'):
        super(MpnnDuvenaud, self).__init__()

        n_layers = len(out_update)

        # Define message 1 & 2
        self.m = nn.ModuleList([MessageFunction('duvenaud') for _ in range(n_layers)])

        # Define Update 1 & 2
        self.u = nn.ModuleList([UpdateFunction('duvenaud', args={'deg': d, 'in': self.m[i].get_out_size(in_n[0], in_n[1]), 'out': out_update[0]}) if i == 0 else
                                UpdateFunction('duvenaud', args={'deg': d, 'in': self.m[i].get_out_size(out_update[i-1], in_n[1]), 'out': out_update[i]}) for i in range(n_layers)])

        # Define Readout
        self.r = ReadoutFunction('duvenaud',
                                 args={'layers': len(self.m) + 1,
                                       'in': [in_n[0] if i == 0 else out_update[i-1] for i in range(n_layers+1)],
                                       'out': hidden_state_readout,
                                       'target': l_target})

        self.type = type

    def forward(self, g, h_in, e, plotter=None):

        h = []
        h.append(h_in)

        # Layer
        for t in range(0, len(self.m)):

            u_args = self.u[t].get_args()

            h_t = Variable(torch.zeros(h_in.size(0), h_in.size(1), u_args['out']).type_as(h[t].data))

            # Apply one layer pass (Message + Update)
            for v in range(0, h_in.size(1)):

                m = self.m[t].forward(h[t][:, v, :], h[t], e[:, v, :])

                # Nodes without edge set message to 0
                m = g[:, v, :, None].expand_as(m) * m

                m = torch.sum(m, 1)

                # Duvenaud
                deg = torch.sum(g[:, v, :].data, 1)

                # Separate degrees
                for i in range(len(u_args['deg'])):
                    ind = deg == u_args['deg'][i]
                    ind = Variable(torch.squeeze(torch.nonzero(torch.squeeze(ind))), volatile=True)

                    opt = {'deg': i}

                    # Update
                    if len(ind) != 0:
                        aux = self.u[t].forward(torch.index_select(h[t], 0, ind)[:, v, :], torch.index_select(m, 0, ind), opt)

                        ind = ind.data.cpu().numpy()
                        for j in range(len(ind)):
                            h_t[ind[j], v, :] = aux[j, :]

            if plotter is not None:
                num_feat = h_t.size(2)
                color = h_t[0,:,:].data.cpu().numpy()
                for i in range(num_feat):
                    plotter(color[:, i], 'layer_' + str(t) + '_element_' + str(i) + '.png')

            h.append(h_t.clone())
        # Readout
        res = self.r.forward(h)
        if self.type == 'classification':
            res = nn.LogSoftmax()(res)
        return res
