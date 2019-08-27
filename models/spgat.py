import torch
import torch.nn as nn
import torch.nn.init as init
import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph
from models.mlp import MLP
import time
import numpy as np

'''test'''
from dgl.data import citation_graph as citegrh

class AGraphATLayer(nn.Module):
    def __init__(self, in_feat, out_feat, embeddim, shortcut=False):
        super(AGraphATLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.embeddim = embeddim
        self.shortcut = shortcut

        self.conv = nn.Sequential(
            nn.Linear(in_feat, out_feat),
            nn.ReLU()
        )
        self.linear_self = nn.Linear(in_feat, out_feat)

        # linear attention
        self.attention = nn.Linear(embeddim, out_feat, bias=False)
        self.batchnorm = nn.BatchNorm1d(out_feat)
        self.act = nn.ReLU()

        # init.xavier_normal_(self.linear)

    def msg_fun(self, edges):
        direction = edges.src['c'] - edges.dst['c']
        attention = self.attention(direction)
        return {'h': attention*self.conv(edges.src['h'])}


    def agg_fun(self, nodes):
        return {'h': torch.sum(nodes.mailbox['h'], dim=1)}

    # sp_embeddings: direction cosine of nodes (batch*n)*embeddim
    def forward(self, graph, feature, sp_embeddings):
        graph.ndata['h'] = feature
        graph.ndata['c'] = sp_embeddings
        graph.update_all(self.msg_fun, self.agg_fun)
        graph.ndata['h'] += self.linear_self(feature)
        if self.in_feat==self.out_feat and self.shortcut:
            graph.ndata['h'] = self.batchnorm(self.act(graph.ndata['h'])) + feature
        else:
            graph.ndata['h'] = self.batchnorm(self.act(graph.ndata['h']))
        return graph.ndata.pop('h')


class AGAT(nn.Module):
    def __init__(self,num_layers, num_mlp_layers, input_dim, hidden_dim, num_classes, embeddim, final_dropout, res_connection,
                 device, k_order):
        super(AGAT, self).__init__()
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embeddim = embeddim

        self.output_dim = num_classes
        self.final_dropout = final_dropout
        self.shortcut = res_connection
        self.device = device

        self._build_model()

    def _build_model(self):
        self.gconv = nn.ModuleList()
        self.classifier = MLP(self.num_mlp_layers, self.hidden_dim, int(self.hidden_dim / 2), self.output_dim)
        self.node_mlp = MLP(2, self.hidden_dim, self.hidden_dim, self.hidden_dim)

        self.gconv.append(AGraphATLayer(self.input_dim, self.hidden_dim, self.embeddim))
        for layer in range(self.num_layers - 1):
            self.gconv.append(AGraphATLayer(self.hidden_dim, self.hidden_dim, self.embeddim,self.shortcut))
        return


    # batch_graphs: list of S2VGraph    bgraph: dgl batched graphs
    def forward(self, batch_graphs):

        # preprocess
        bgraph = dgl.batch([graph.to_dgl() for graph in batch_graphs])
        feature = torch.cat([graph.node_features for graph in batch_graphs], 0).to(self.device)
        sp_embeddings = torch.cat([graph.shortest_path for graph in batch_graphs], 0).to(self.device)

        logits = 0
        for layer in range(self.num_layers):
            feature = self.gconv[layer](bgraph, feature, sp_embeddings)
            bgraph.ndata['h'] = feature
            logits += self.classifier(F.dropout(dgl.max_nodes(bgraph, 'h'), self.final_dropout))

        # bgraph.ndata['h'] = self.node_mlp(feature)
        # logits = F.dropout(self.classifier(dgl.sum_nodes(bgraph, 'h')),self.final_dropout)

        return logits


class SPGConvLayer(nn.Module):
    def __init__(self, in_feat, out_feat, k_order=1, shortcut=False):
        super(SPGConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.k_order = k_order
        self.shortcut = shortcut

        # init model
        self.linear = nn.Parameter(torch.Tensor(k_order + 1, in_feat, out_feat))    # self loop!
        self.mlp = nn.Sequential(
            nn.Linear(out_feat, out_feat),
            nn.ReLU(),
            nn.BatchNorm1d(out_feat)
        )

        init.xavier_normal_(self.linear)

    def msg_fun(self, edges):
        return {'h': torch.einsum('bij,bi->bj',[self.linear[edges.data['order']],edges.src['h']])}  # src (batch*n)*dim


    def agg_fun(self, nodes):
        return {'h': torch.sum(nodes.mailbox['h'], dim=1)}


    def forward(self, graph, feature, sp_embeddings):
        graph.ndata['h'] = feature
        graph.update_all(self.msg_fun, self.agg_fun)
        graph.ndata['h'] = self.mlp(graph.ndata['h'])
        if self.in_feat==self.out_feat and self.shortcut:
            graph.ndata['h'] += feature
        return graph.ndata.pop('h')


class SPGNN(nn.Module):
    def __init__(self,num_layers, num_mlp_layers, input_dim, hidden_dim, num_classes, embeddim, final_dropout, res_connection,
                 device, k_order=1):
        super(SPGNN, self).__init__()
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embeddim = embeddim
        self.k_order = k_order

        self.output_dim = num_classes
        self.final_dropout = final_dropout
        self.shortcut = res_connection
        self.device = device

        #inside options
        self.jump = False

        self._build_model()

    def _build_model(self):
        self.gconv = nn.ModuleList()
        self.classifier = MLP(self.num_mlp_layers, self.hidden_dim, int(self.hidden_dim/2), self.output_dim)
        self.embedding_layer = nn.Linear(self.input_dim, self.hidden_dim, bias=True)    #transform input 0 to feature

        for layer in range(self.num_layers):
            self.gconv.append(SPGConvLayer(self.hidden_dim, self.hidden_dim, self.k_order))

        return




    def forward(self, batch_graphs):
        # preprocess
        bgraph = dgl.batch([graph.dgl_graph for graph in batch_graphs])
        feature = torch.cat([graph.node_features for graph in batch_graphs], 0).to(self.device)
        sp_embeddings = torch.cat([graph.shortest_path for graph in batch_graphs], 0).to(self.device)

        logits = 0
        feature = self.embedding_layer(feature)
        for layer in range(self.num_layers):
            feature = self.gconv[layer](bgraph, feature, sp_embeddings)
            bgraph.ndata['h'] = feature

            if self.jump:
                logits += self.classifier(F.dropout(dgl.sum_nodes(bgraph,'h'), self.final_dropout))

        bgraph.ndata['h'] = feature
        logits += self.classifier(F.dropout(dgl.sum_nodes(bgraph, 'h'), self.final_dropout))

        return logits



class ASPGATLayer(nn.Module):
    def __init__(self, in_feat, out_feat, embeded_dim, k_order=1, shortcut=False):
        super(ASPGATLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.k_order = k_order
        self.shortcut = shortcut
        self.embeded_dim = embeded_dim

        # init model
        self.linear = nn.Parameter(torch.Tensor(k_order + 1, in_feat, out_feat))    # self loop!
        self.mlp = nn.Sequential(
            nn.Linear(out_feat, out_feat),
            nn.ReLU(),
            nn.BatchNorm1d(out_feat)
        )
        self.attention_layer = nn.Linear(self.embeded_dim, self.out_feat, bias=False)
        self.bn = nn.BatchNorm1d(out_feat)
        init.xavier_normal_(self.linear)


    def edge_att(self, edges):
        attention = torch.sum(self.attention_layer(edges.src['c']-edges.dst['c']) * edges.src['h'], dim=1)
        invdist = 1/(torch.norm(edges.src['c']-edges.dst['c'], 2, dim=1)**2 + 1)
        return {'e': attention, 'd': invdist}


    def msg_fun(self, edges):
        # directions = (edges.src['c']-edges.dst['c'])/torch.norm(edges.src['c']-edges.dst['c'],2,dim=1,keepdim=True)
        # directions = F.normalize(edges.src['c']-edges.dst['c']+1e-3, p=2, dim=1)
        # attention = self.attention_layer(directions)
        # msg = attention*self.bn(F.relu(torch.einsum('bij,bi->bj',[self.linear[edges.data['order']],edges.src['h']])))
        msg = torch.einsum('bij,bi->bj',[self.linear[edges.data['order']],edges.src['h']])
        return {'h': msg, 'e': edges.data['e'], 'd': edges.data['d']}  # src (batch*n)*dim


    def agg_fun(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        return {'h': torch.sum((alpha * nodes.mailbox['d']).unsqueeze(-1) * nodes.mailbox['h'], dim=1)}


    def forward(self, graph, feature, sp_embeddings):
        graph.ndata['h'] = feature
        graph.ndata['c'] = sp_embeddings
        graph.apply_edges(self.edge_att)    # when full attention is needed
        graph.update_all(self.msg_fun, self.agg_fun)
        graph.ndata['h'] = self.mlp(graph.ndata['h'])
        if self.in_feat==self.out_feat and self.shortcut:
            graph.ndata['h'] += feature
        return graph.ndata.pop('h')


class ASPGAT(nn.Module):
    def __init__(self,num_layers, num_mlp_layers, input_dim, hidden_dim, num_classes, embeddim, final_dropout, res_connection,
                 device, k_order=1):
        super(ASPGAT, self).__init__()
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embeddim = embeddim
        self.k_order = k_order

        self.output_dim = num_classes
        self.final_dropout = final_dropout
        self.shortcut = res_connection
        self.device = device

        #inside options
        self.jump = True

        self._build_model()

    def _build_model(self):
        self.gconv = nn.ModuleList()
        self.classifier = MLP(self.num_mlp_layers, self.hidden_dim, int(self.hidden_dim/2), self.output_dim)
        self.embedding_layer = nn.Linear(self.input_dim, self.hidden_dim, bias=True)    #transform input 0 to feature

        for layer in range(self.num_layers):
            self.gconv.append(ASPGATLayer(self.hidden_dim, self.hidden_dim, self.embeddim, self.k_order))

        return


    def forward(self, batch_graphs):
        # preprocess
        bgraph = dgl.batch([graph.dgl_graph for graph in batch_graphs])
        feature = torch.cat([graph.node_features for graph in batch_graphs], 0).to(self.device)
        sp_embeddings = torch.cat([graph.shortest_path for graph in batch_graphs], 0).to(self.device)

        logits = 0
        feature = self.embedding_layer(feature)
        for layer in range(self.num_layers):
            feature = self.gconv[layer](bgraph, feature, sp_embeddings)
            bgraph.ndata['h'] = feature

            if self.jump:
                logits += self.classifier(F.dropout(dgl.sum_nodes(bgraph,'h'), self.final_dropout))

        bgraph.ndata['h'] = feature
        logits += self.classifier(F.dropout(dgl.sum_nodes(bgraph, 'h'), self.final_dropout))

        return logits