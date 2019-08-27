import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import MLP
import time


class gcnn_new(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, feature_dim, hidden_dim, num_classes,
                 final_dropout, graph_pooling_type, device, embedded_dim):
        super(gcnn_new, self).__init__()
        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.graph_pooling_type = graph_pooling_type
        self.feature_dim = feature_dim
        self.embedded_dim = embedded_dim
        self.mlps = torch.nn.ModuleList()
        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, feature_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(feature_dim, num_classes))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, num_classes))
        self.conv_kernel = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.conv_kernel.append(MLP(num_mlp_layers, embedded_dim, hidden_dim, feature_dim))
                self.batch_norms.append(nn.BatchNorm1d(feature_dim))
            else:
                self.conv_kernel.append(MLP(num_mlp_layers, embedded_dim, hidden_dim, hidden_dim))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def get_new_feat(self, dist, feature, num):
        dist = dist.view(1, -1)
        # print(dist.size())
        # a = time.clock()
        h = self.conv_kernel[num](dist)
        # print("conv_kernel MLP time=", time.clock()-a)#10-5
        h.mul(feature)
        return h

    def gather_new_feats(self, distance, features, num):
        # a = time.clock()
        dist = self.conv_kernel[num](distance)
        return dist.mul(features).sum(0)
        # new_feat = torch.zeros(features.size()[1]).to(self.device)
        # for i, dist in enumerate(distance):
        #    new_feat = new_feat + self.get_new_feat(dist, features[i], num)
        # print("gather_new_feats_time=",time.clock()-a)#5e-4 -> 1e-4
        # return new_feat

    def Aggregate(self, batch_graph, feature_concat, start_idx, num):
        # a = time.clock()
        h = torch.zeros(torch.Size([start_idx[-1], self.hidden_dim]))  # return tensor
        # 1e03 nodes
        for i in range(len(start_idx) - 1):
            for j in range(start_idx[i], start_idx[i + 1]):
                '''distance = batch_graph[i].shortest_path[batch_graph[i].
                                             neighbors[j-start_idx[i]]
                                             ]-batch_graph[i].shortest_path[j-start_idx[i]]'''
                features = feature_concat[np.array(batch_graph[i].neighbors[j - start_idx[i]]) + start_idx[i]]
                # print(time.clock()-a) # 1e-4
                a = time.clock()
                f = (features.sum(0) + feature_concat[j])
                print(time.clock() - a)
                h[j] = self.mlps[num](f)
                '''self.gather_new_feats(distance, features, num)'''
        # print("Aggregate_time=",time.clock()-a)# 0.8s -> 0.3s
        return h

    def preprocess(self, batch_graph):
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
        return start_idx

    def preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1. / len(graph.g)] * len(graph.g))

            else:
                ###sum pooling
                elem.extend([1] * len(graph.g))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

        return graph_pool.to(self.device)

    '''def forward(self, batch_graph):
        start_idx = self.preprocess(batch_graph)
        print(start_idx[-1])
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        h = X_concat
        hidden_rep = [X_concat]
        for i in range(self.num_layers-1):
            h = self.Aggregate(batch_graph, h, start_idx, i)
            hidden_rep.append(h)
        graph_pool = self.preprocess_graphpool(batch_graph)
        score_over_layer = 0
        for layer, h in enumerate(hidden_rep):
            pooled_h = torch.spmm(graph_pool, h)
            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout, training = self.training)
        return score_over_layer'''

    def forward(self, batch_graph):  # 1.88s
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        h = X_concat
        hidden_rep = [X_concat]
        # a = time.clock()
        Neighbor_concat = torch.cat([graph.shortest_path for graph in batch_graph], 0).to(self.device)
        # print("time for prepare data:",time.clock()-a)#for rdt data: 0.38s
        # 0.03s
        graph_pool = self.preprocess_graphpool(batch_graph)
        Adj_block = self.__preprocess_graphpool(batch_graph)  # 0.02s
        V2E_block = self.get_V2E(batch_graph)
        # print(V2E_block.size())
        # print(X_concat.size())
        g = torch.spmm(V2E_block, X_concat)
        # print(g.size())
        Neighbor_concat = torch.spmm(self.get_V2E_2(batch_graph), Neighbor_concat)
        if abs(Neighbor_concat).max() > 1:
            print("error!")
        # print(Neighbor_concat)

        for i in range(self.num_layers - 1):
            weight = self.conv_kernel[i](Neighbor_concat)  # 0.006s
            weight = weight.mul(g)  # 1e-4 s
            weight = weight.sum(1).view(-1, 1)  # 1e-4 s
            weight = weight * (1.0 * h.size()[0] / weight.sum())
            g = g * weight  # 6.75e-5 s
            h = self.next_layer(h, g, i, Adj_block)  # 0.008s
            g = torch.spmm(V2E_block, h)
            # 0.42s
            hidden_rep.append(h)
        score_over_layer = 0
        # print("time for cycle:", time.clock()-a)#1.47s

        # perform pooling over all nodes in each graph in every layer
        for layer, H in enumerate(hidden_rep):
            pooled_h = torch.spmm(graph_pool, H)
            score_over_layer += F.dropout(torch.sigmoid(self.linears_prediction[layer](pooled_h)), self.final_dropout,
                                          training=self.training)
        # print(time.clock()-a) #final cycle: 0.02s
        return score_over_layer

    def __preprocess_neighbors_sumpool(self, batch_graph, Adj_block_elem):
        ###create block diagonal sparse matrix

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])  # 可以预处理
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        # Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        # Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1], start_idx[-1]]))

        return Adj_block.to(self.device)

    def __preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)
        start_idx = [0]
        k = 0
        for i, graph in enumerate(batch_graph):
            for j in range(len(graph.g)):
                start_idx.append(start_idx[k] + len(graph.neighbors[j]))
                k += 1

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            for j in range(len(graph.g)):
                ###average pooling
                if len(graph.neighbors[j]):
                    elem.extend([len(graph.neighbors[j])] * len(graph.neighbors[j]))
            '''else:
            ###sum pooling
                elem.extend([1]*len(graph.g))'''
        for i in range(len(start_idx) - 1):
            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(start_idx) - 1, start_idx[-1]]))

        return graph_pool.to(self.device)

    def get_V2E(self, batch_graph):
        idx = []
        elem = []
        k = 0
        v = 0
        for graph in batch_graph:
            for j in range(len(graph.g)):
                for l in graph.neighbors[j]:
                    idx.extend([[k, l + v]])  # !!!!!!!!!!!!!
                    k += 1
                elem.extend([1] * len(graph.neighbors[j]))
            v += len(graph.g)
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        V2E_block = torch.sparse.FloatTensor(idx, elem, torch.Size([k, v]))

        return V2E_block.to(self.device)

    def get_V2E_2(self, batch_graph):
        idx = []
        elem = []
        k = 0
        v = 0
        for graph in batch_graph:
            for j in range(len(graph.g)):
                for l in graph.neighbors[j]:
                    idx.extend([[k, j + v]])
                    idx.extend([[k, l + v]])
                    k += 1
                elem.extend([-1, 1] * len(graph.neighbors[j]))
            v += len(graph.g)
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        V2E_block = torch.sparse.FloatTensor(idx, elem, torch.Size([k, v]))

        return V2E_block.to(self.device)

    def next_layer(self, h, g, layer, Adj_block):
        pooled = torch.spmm(Adj_block, g)
        pooled = pooled + h
        return self.mlps[layer](pooled)

