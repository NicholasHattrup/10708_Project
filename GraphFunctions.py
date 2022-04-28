import os, argparse

import numpy as np
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import torch.nn.functional as F

from NNet import NNet

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat" 


class MessageFunction(nn.Module):

    # Constructor
    def __init__(self, message_def='mpnn', args={}):
        super(MessageFunction, self).__init__()
        self.m_definition = ''
        self.m_function = None
        self.args = {}
        self.__set_message(message_def, args)

    # Message from h_v to h_w through e_vw
    def forward(self, h_v, h_w, e_vw, args=None):
        return self.m_function(h_v, h_w, e_vw, args)

    # Set a message function
    def __set_message(self, message_def, args={}):
        self.m_definition = message_def.lower()

        self.m_function = {
                    'duvenaud':         self.m_duvenaud,
                    'ggnn':             self.m_ggnn,
                    'intnet':           self.m_intnet,
                    'mpnn':             self.m_mpnn
                }.get(self.m_definition, None)

        if self.m_function is None:
            print('WARNING!: Message Function has not been set correctly\n\tIncorrect definition ' + message_def)
            quit()

        init_parameters = {
            'duvenaud': self.init_duvenaud,
            'ggnn':     self.init_ggnn,
            'intnet':   self.init_intnet,
            'mpnn':     self.init_mpnn
        }.get(self.m_definition, lambda x: (nn.ParameterList([]), nn.ModuleList([]), {}))

        self.learn_args, self.learn_modules, self.args = init_parameters(args)

        self.m_size = {
                'duvenaud':     self.out_duvenaud,
                'ggnn':         self.out_ggnn,
                'intnet':       self.out_intnet,
                'mpnn':         self.out_mpnn
            }.get(self.m_definition, None)

    # Get the name of the used message function
    def get_definition(self):
        return self.m_definition

    # Get the message function arguments
    def get_args(self):
        return self.args

    # Get Output size
    def get_out_size(self, size_h, size_e, args=None):
        return self.m_size(size_h, size_e, args)

    # Definition of various state of the art message functions
    
    # Duvenaud et al. (2015), Convolutional Networks for Learning Molecular Fingerprints
    def m_duvenaud(self, h_v, h_w, e_vw, args):
        m = torch.cat([h_w, e_vw], 2)
        return m

    def out_duvenaud(self, size_h, size_e, args):
        return size_h + size_e

    def init_duvenaud(self, params):
        learn_args = []
        learn_modules = []
        args = {}
        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args

    # Li et al. (2016), Gated Graph Neural Networks (GG-NN)
    def m_ggnn(self, h_v, h_w, e_vw, opt={}):

        m = Variable(torch.zeros(h_w.size(0), h_w.size(1), self.args['out']).type_as(h_w.data))

        for w in range(h_w.size(1)):
            if torch.nonzero(e_vw[:, w, :].data).size():
                for i, el in enumerate(self.args['e_label']):
                    ind = (el == e_vw[:,w,:]).type_as(self.learn_args[0][i])

                    parameter_mat = self.learn_args[0][i][None, ...].expand(h_w.size(0), self.learn_args[0][i].size(0),
                                                                            self.learn_args[0][i].size(1))

                    m_w = torch.transpose(torch.bmm(torch.transpose(parameter_mat, 1, 2),
                                                                        torch.transpose(torch.unsqueeze(h_w[:, w, :], 1),
                                                                                        1, 2)), 1, 2)
                    m_w = torch.squeeze(m_w)
                    m[:,w,:] = ind.expand_as(m_w)*m_w
        return m

    def out_ggnn(self, size_h, size_e, args):
        return self.args['out']

    def init_ggnn(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        args['e_label'] = params['e_label']
        args['in'] = params['in']
        args['out'] = params['out']

        # Define a parameter matrix A for each edge label.
        learn_args.append(nn.Parameter(torch.randn(len(params['e_label']), params['in'], params['out'])))

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args

    # Battaglia et al. (2016), Interaction Networks
    def m_intnet(self, h_v, h_w, e_vw, args):
        m = torch.cat([h_v[:, None, :].expand_as(h_w), h_w, e_vw], 2)
        b_size = m.size()

        m = m.view(-1, b_size[2])

        m = self.learn_modules[0](m)
        m = m.view(b_size[0], b_size[1], -1)
        return m

    def out_intnet(self, size_h, size_e, args):
        return self.args['out']

    def init_intnet(self, params):
        learn_args = []
        learn_modules = []
        args = {}
        args['in'] = params['in']
        args['out'] = params['out']
        learn_modules.append(NNet(n_in=params['in'], n_out=params['out']))
        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args

    # Gilmer et al. (2017), Neural Message Passing for Quantum Chemistry
    def m_mpnn(self, h_v, h_w, e_vw, opt={}):
        # Matrices for each edge
        edge_output = self.learn_modules[0](e_vw)
        edge_output = edge_output.view(-1, self.args['out'], self.args['in'])

        h_w_rows = h_w[..., None].expand(h_w.size(0), h_w.size(1), h_v.size(1)).contiguous()

        h_w_rows = h_w_rows.view(-1, self.args['in'])

        h_multiply = torch.bmm(edge_output, torch.unsqueeze(h_w_rows,2))

        m_new = torch.squeeze(h_multiply)

        return m_new

    def out_mpnn(self, size_h, size_e, args):
        return self.args['out']

    def init_mpnn(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        args['in'] = params['in']
        args['out'] = params['out']

        # Define a parameter matrix A for each edge label.
        learn_modules.append(NNet(n_in=params['edge_feat'], n_out=(params['in']*params['out'])))

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args


class UpdateFunction(nn.Module):

    # Constructor
    def __init__(self, update_def='mpnn', args={}):
        super(UpdateFunction, self).__init__()
        self.u_definition = ''
        self.u_function = None
        self.args = {}
        self.__set_update(update_def, args)

    # Update node hv given message mv
    def forward(self, h_v, m_v, opt={}):
        return self.u_function(h_v, m_v, opt)

    # Set update function
    def __set_update(self, update_def, args):
        self.u_definition = update_def.lower()

        self.u_function = {
                    'duvenaud':         self.u_duvenaud,
                    'ggnn':             self.u_ggnn,
                    'intnet':           self.u_intnet,
                    'mpnn':             self.u_mpnn
                }.get(self.u_definition, None)

        if self.u_function is None:
            print('WARNING!: Update Function has not been set correctly\n\tIncorrect definition ' + update_def)

        init_parameters = {
            'duvenaud':         self.init_duvenaud,
            'ggnn':             self.init_ggnn,
            'intnet':           self.init_intnet,
            'mpnn':             self.init_mpnn
        }.get(self.u_definition, lambda x: (nn.ParameterList([]), nn.ModuleList([]), {}))

        self.learn_args, self.learn_modules, self.args = init_parameters(args)

    # Get the name of the used update function
    def get_definition(self):
        return self.u_definition

    # Get the update function arguments
    def get_args(self):
        return self.args

    ## Definition of various state of the art update functions

    # Duvenaud
    def u_duvenaud(self, h_v, m_v, opt):

        param_sz = self.learn_args[0][opt['deg']].size()
        parameter_mat = torch.t(self.learn_args[0][opt['deg']])[None, ...].expand(m_v.size(0), param_sz[1], param_sz[0])

        aux = torch.bmm(parameter_mat, torch.transpose(m_v, 1, 2))

        return torch.transpose(torch.nn.Sigmoid()(aux), 1, 2)

    def init_duvenaud(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        # Filter degree 0 (the message will be 0 and therefore there is no update
        args['deg'] = [i for i in params['deg'] if i!=0]
        args['in'] = params['in']
        args['out'] = params['out']

        # Define a parameter matrix H for each degree.
        learn_args.append(torch.nn.Parameter(torch.randn(len(args['deg']), args['in'], args['out'])))

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args

    # GG-NN, Li et al.
    def u_ggnn(self, h_v, m_v, opt={}):
        h_v.contiguous()
        m_v.contiguous()
        h_new = self.learn_modules[0](torch.transpose(m_v, 0, 1), torch.unsqueeze(h_v, 0))[0]  # 0 or 1???
        return torch.transpose(h_new, 0, 1)

    def init_ggnn(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        args['in_m'] = params['in_m']
        args['out'] = params['out']

        # GRU
        learn_modules.append(nn.GRU(params['in_m'], params['out']))

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args

    # Battaglia et al. (2016), Interaction Networks
    def u_intnet(self, h_v, m_v, opt):
        if opt['x_v'].ndimension():
            input_tensor = torch.cat([h_v, opt['x_v'], torch.squeeze(m_v)], 1)
        else:
            input_tensor = torch.cat([h_v, torch.squeeze(m_v)], 1)

        return self.learn_modules[0](input_tensor)

    def init_intnet(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        args['in'] = params['in']
        args['out'] = params['out']

        learn_modules.append(NNet(n_in=params['in'], n_out=params['out']))

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args

    def u_mpnn(self, h_v, m_v, opt={}):
        h_in = h_v.view(-1,h_v.size(2))
        m_in = m_v.view(-1,m_v.size(2))
        h_new = self.learn_modules[0](m_in[None,...],h_in[None,...])[0] # 0 or 1???
        return torch.squeeze(h_new).view(h_v.size())

    def init_mpnn(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        args['in_m'] = params['in_m']
        args['out'] = params['out']

        # GRU
        learn_modules.append(nn.GRU(params['in_m'], params['out']))

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args


class ReadoutFunction(nn.Module):

    # Constructor
    def __init__(self, readout_def='mpnn', args={}):
        super(ReadoutFunction, self).__init__()
        self.r_definition = ''
        self.r_function = None
        self.args = {}
        self.__set_readout(readout_def, args)

    # Readout graph given node values at las layer
    def forward(self, h_v):
        return self.r_function(h_v)

    # Set a readout function
    def __set_readout(self, readout_def, args):
        self.r_definition = readout_def.lower()

        self.r_function = {
                    'duvenaud': self.r_duvenaud,
                    'ggnn':     self.r_ggnn,
                    'intnet':   self.r_intnet,
                    'mpnn':     self.r_mpnn
                }.get(self.r_definition, None)

        if self.r_function is None:
            print('WARNING!: Readout Function has not been set correctly\n\tIncorrect definition ' + readout_def)
            quit()

        init_parameters = {
            'duvenaud': self.init_duvenaud,
            'ggnn':     self.init_ggnn,
            'intnet':   self.init_intnet,
            'mpnn':     self.init_mpnn
        }.get(self.r_definition, lambda x: (nn.ParameterList([]), nn.ModuleList([]), {}))

        self.learn_args, self.learn_modules, self.args = init_parameters(args)

    # Get the name of the used readout function
    def get_definition(self):
        return self.r_definition

    ## Definition of various state of the art update functions

    # Duvenaud
    def r_duvenaud(self, h):
        # layers
        aux = []
        for l in range(len(h)):
            param_sz = self.learn_args[l].size()
            parameter_mat = torch.t(self.learn_args[l])[None, ...].expand(h[l].size(0), param_sz[1],
                                                                                      param_sz[0])

            aux.append(torch.transpose(torch.bmm(parameter_mat, torch.transpose(h[l], 1, 2)), 1, 2))

            for j in range(0, aux[l].size(1)):
                # Mask whole 0 vectors
                aux[l][:, j, :] = nn.Softmax()(aux[l][:, j, :].clone())*(torch.sum(aux[l][:, j, :] != 0, 1) > 0).expand_as(aux[l][:, j, :]).type_as(aux[l])

        aux = torch.sum(torch.sum(torch.stack(aux, 3), 3), 1)
        return self.learn_modules[0](torch.squeeze(aux))

    def init_duvenaud(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        args['out'] = params['out']

        # Define a parameter matrix W for each layer.
        for l in range(params['layers']):
            learn_args.append(nn.Parameter(torch.randn(params['in'][l], params['out'])))

        # learn_modules.append(nn.Linear(params['out'], params['target']))

        learn_modules.append(NNet(n_in=params['out'], n_out=params['target']))
        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args

    # GG-NN, Li et al.
    def r_ggnn(self, h):

        aux = Variable( torch.Tensor(h[0].size(0), self.args['out']).type_as(h[0].data).zero_() )
        # For each graph
        for i in range(h[0].size(0)):
            nn_res = nn.Sigmoid()(self.learn_modules[0](torch.cat([h[0][i,:,:], h[-1][i,:,:]], 1)))*self.learn_modules[1](h[-1][i,:,:])

            # Delete virtual nodes
            nn_res = (torch.sum(h[0][i,:,:],1).expand_as(nn_res)>0).type_as(nn_res)* nn_res

            aux[i,:] = torch.sum(nn_res,0)

        return aux

    def init_ggnn(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        # i
        learn_modules.append(NNet(n_in=2*params['in'], n_out=params['target']))

        # j
        learn_modules.append(NNet(n_in=params['in'], n_out=params['target']))

        args['out'] = params['target']

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args


    # Battaglia et al. (2016), Interaction Networks
    def r_intnet(self, h):

        aux = torch.sum(h[-1],1)

        return self.learn_modules[0](aux)

    def init_intnet(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        learn_modules.append(NNet(n_in=params['in'], n_out=params['target']))

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args

    def r_mpnn(self, h):

        aux = Variable( torch.Tensor(h[0].size(0), self.args['out']).type_as(h[0].data).zero_() )
        # For each graph
        for i in range(h[0].size(0)):
            nn_res = nn.Sigmoid()(self.learn_modules[0](torch.cat([h[0][i,:,:], h[-1][i,:,:]], 1)))*self.learn_modules[1](h[-1][i,:,:])

            # Delete virtual nodes
            nn_res = (torch.sum(h[0][i,:,:],1)[...,None].expand_as(nn_res)>0).type_as(nn_res)* nn_res

            aux[i,:] = torch.sum(nn_res,0)

        return aux

    def init_mpnn(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        # i
        learn_modules.append(NNet(n_in=2*params['in'], n_out=params['target']))

        # j
        learn_modules.append(NNet(n_in=params['in'], n_out=params['target']))

        args['out'] = params['target']

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args
