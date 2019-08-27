import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchnet.meter import AverageValueMeter
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
# from tqdm import tqdm
import time
import os
from util import load_data, separate_data, load_pkl

# models
from models.graphcnn import GraphCNN
from models import AGAT, AGCN, GIN, SPGNN, ASPGAT


work_path = os.path.dirname(__file__)
log_path = work_path + '/logs/' + time.strftime('%m%d_%H_%M')
criterion = nn.CrossEntropyLoss()

def train(args, model, writer, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch

    # pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in range(total_iters):
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        #compute loss
        loss = criterion(output, labels)

        if args.use_tb:
            writer.add_scalar('training_loss',loss.cpu(),(pos+1)/50+epoch)
        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        # pbar.set_description('epoch: %d' % (epoch))

    for param in optimizer.param_groups:
        lr = param['lr']
    average_loss = loss_accum/total_iters
    print("loss training:{}  epoch:{}  lr:{}".format(average_loss, epoch, lr))
    
    return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 16):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(args, model, writer, device, train_graphs, test_graphs, epoch):
    model.eval()
    with torch.no_grad():
        output = pass_data_iteratively(model, train_graphs)
        pred = output.max(1, keepdim=True)[1]
        labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
        correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
        acc_train = correct / float(len(train_graphs))

        output = pass_data_iteratively(model, test_graphs)
        pred = output.max(1, keepdim=True)[1]
        labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
        test_loss = criterion(output, labels)
        correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
        acc_test = correct / float(len(test_graphs))

        if args.use_tb:
            writer.add_scalar('train_acc',acc_train,epoch)
            writer.add_scalar('test_acc',acc_test,epoch)

        print("accuracy train: %f test: %f loss test: %f" % (acc_train, acc_test, test_loss))

    return acc_train, acc_test

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="PROTEINS",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate (default: 0.01)')
    # parser.add_argument('--lr_decay', type=float, default=0.8,
    #                     help='learning rate decay with epochs')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--embedded_dim',type=int, default=4,
                        help='number of the embedding dimension using shortest path length to sources')
    parser.add_argument('--orders', type=int, default=1,
                        help='number of neighbors order use in conv layers')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--res_connection',type=bool, default=False,
                        help='whether add shortcut or res_connection')
    parser.add_argument('--degree_as_tag', type=bool, default=True,
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    parser.add_argument('--use_tb', type=bool, default=False,
                        help='use tensorboard to record loss')
    args = parser.parse_args()

    #       SOTA Performance Hyperparameters
    #   dataset    model    lr   batchsize   epoch   layers    dropout   orders    test_acc     train_acc
    #   PROTEINS   AGAT    5e-3    32        10~20     3        0.5        1        0.794           0.82
    #   PROTEINS   SPGNN   1e-3    32         ~30      2        0.5        2        0.771(15epochs)  ~0.77
    #   PROTEINS   SPGNN   1e-3    32         ~80      2        0.5        2        0.786            0.80
    #
    use_default = False
    if not use_default:
        args.dataset = 'PTC'
        args.device = 0
        args.batch_size =32
        args.lr = 1e-2
        args.num_layers = 3
        args.num_mlp_layers = 3
        args.embedded_dim = 6
        args.hidden_dim = 32
        args.final_dropout = 0.5
        args.orders = 3
        args.degree_as_tag = True
        args.res_connection = True
        args.use_tb = True
        args.iters_per_epoch = 30

    model_ = AGCN

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    graphs, num_classes = load_data(args.dataset, args.degree_as_tag, args.embedded_dim, )
    # graphs, num_classes = load_pkl(args.dataset, args.degree_as_tag, args.embedded_dim)

    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)


    model = model_(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.embedded_dim, args.final_dropout, args.res_connection, device, args.orders).to(device)
    # model = AGAT(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.embedded_dim, args.final_dropout, args.res_connection, device).to(device)
    # model = GIN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.res_connection, device).to(device)

    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(model)
    writer = SummaryWriter(log_dir=log_path) if args.use_tb else None

    count_step_size = 15
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.75)


    acc_test_meter = AverageValueMeter()
    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        avg_loss = train(args, model, writer, device, train_graphs, optimizer, epoch)
        acc_train, acc_test = test(args, model, writer, device, train_graphs, test_graphs, epoch)

        acc_test_meter.add(acc_test)

        if epoch%count_step_size == 0:
            print('------last {} epochs test acc:{}'.format(count_step_size, acc_test_meter.value()[0]))
            acc_test_meter.reset()


        if not args.filename == "":
            with open(args.filename, 'w') as f:
                f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
                f.write("\n")
        print("")

        # print(model.eps)
    

if __name__ == '__main__':
    main()
