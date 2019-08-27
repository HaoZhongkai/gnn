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


# a single GCN layer
class GraphConvLayer(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(GraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        # init model
        self.linear = nn.Linear(in_feat, out_feat, bias=False)
        self.batchnorm = nn.BatchNorm1d(out_feat)
        self.act = nn.ReLU()

    def msg_fun(self, edges):
        return {'h': self.linear(edges.src['h'])}   # src (batch*n)*dim

    def agg_fun(self, nodes):
        return {'h': torch.sum(nodes.mailbox['h'], dim=1)}

    def forward(self, graph, feature):
        graph.ndata['h'] = feature
        graph.update_all(self.msg_fun, self.agg_fun)
        return self.act(self.batchnorm(graph.ndata.pop('h')))


class AGraphConvLayer(nn.Module):
    def __init__(self, in_feat, out_feat, embeddim, shortcut=False):
        super(AGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.embeddim = embeddim
        self.shortcut = shortcut

        self.linear = nn.Parameter(torch.Tensor(embeddim, in_feat, out_feat))
        self.batchnorm = nn.BatchNorm1d(out_feat)
        self.act = nn.ReLU()

        init.xavier_normal_(self.linear)

    def msg_fun(self, edges):
        direction = F.normalize(edges.src['c'] - edges.dst['c'], p=2, dim=1)
        return {'h': torch.einsum('bi,ijk,bj->bk', [direction, self.linear, edges.src['h']])}

    def agg_fun(self, nodes):
        return {'h': torch.sum(nodes.mailbox['h'], dim=1)}

    # sp_embeddings: direction cosine of nodes (batch*n)*embeddim
    def forward(self, graph, feature, sp_embeddings):
        graph.ndata['h'] = feature
        graph.ndata['c'] = sp_embeddings
        graph.update_all(self.msg_fun, self.agg_fun)
        if self.in_feat==self.out_feat and self.shortcut:
            graph.ndata['h'] = self.batchnorm(self.act(graph.ndata['h'])) + feature
        else:
            graph.ndata['h'] = self.batchnorm(self.act(graph.ndata['h']))
        return graph.ndata.pop('h')


class GIN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, num_classes, embedded_dim, final_dropout, res_connection,
                 device, k_order):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.output_dim = num_classes
        self.final_dropout = final_dropout
        self.shortcut = res_connection
        self.device = device

        self._build_model()

    def _build_model(self):
        self.gconv = nn.ModuleList()
        self.classifier = MLP(self.num_mlp_layers, self.hidden_dim, int(self.hidden_dim / 2), self.output_dim)

        self.gconv.append(GraphConvLayer(self.input_dim, self.hidden_dim))
        for layer in range(self.num_layers - 1):
            self.gconv.append(GraphConvLayer(self.hidden_dim, self.hidden_dim))
        return

    # batch_graphs: list of S2VGraph    bgraph: dgl batched graphs
    def forward(self, batch_graphs):

        # preprocess
        bgraph = dgl.batch([graph.to_dgl() for graph in batch_graphs])
        feature = torch.cat([graph.node_features for graph in batch_graphs], 0).to(self.device)

        for layer in range(self.num_layers):
            feature = self.gconv[layer](bgraph, feature)

        bgraph.ndata['h'] = feature
        logits = self.classifier(dgl.sum_nodes(bgraph, 'h'))

        return logits


class AGCN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, num_classes, embeddim, final_dropout, res_connection,
                 device, k_order):
        super(AGCN, self).__init__()
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

        self.gconv.append(AGraphConvLayer(self.input_dim, self.hidden_dim, self.embeddim))
        for layer in range(self.num_layers - 1):
            self.gconv.append(AGraphConvLayer(self.hidden_dim, self.hidden_dim, self.embeddim,self.shortcut))
        return

    # preprocess sp vector
    def _preprocess(self, batch_graphs):
        pass

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
            logits += self.classifier(F.dropout(dgl.sum_nodes(bgraph, 'h'), self.final_dropout))

        # bgraph.ndata['h'] = self.node_mlp(feature)
        # logits = F.dropout(self.classifier(dgl.sum_nodes(bgraph, 'h')),self.final_dropout)

        return logits


def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.ByteTensor(data.train_mask)
    g = data.graph
    # add self loop
    g.remove_edges_from(g.selfloop_edges())
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, mask


'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(1433, 16)
        self.gcn2 = GCN(16, 7)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x


net = GCN(2,2,1433,64,7,0.5,False,torch.device("cuda:0"))
print(net)

g, features, labels, mask = load_cora_data()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
dur = []
for epoch in range(30):
    if epoch >= 3:
        t0 = time.time()

    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[mask], labels[mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
        epoch, loss.item(), np.mean(dur)))
'''
