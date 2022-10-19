import torch
import torch.nn as nn
from dgl.nn.pytorch import conv as dgl_conv
from sklearn.metrics import classification_report, balanced_accuracy_score, \
    precision_recall_fscore_support, accuracy_score


class GraphSAGEModel(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_dim,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGEModel, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(dgl_conv.SAGEConv(in_feats, n_hidden, aggregator_type,
                                         feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(dgl_conv.SAGEConv(n_hidden, n_hidden, aggregator_type,
                                             feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(dgl_conv.SAGEConv(n_hidden, out_dim, aggregator_type,
                                         feat_drop=dropout, activation=None))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h
    
class NodeClassification(nn.Module):
    def __init__(self, gconv_model, n_hidden, n_classes):
        super(NodeClassification, self).__init__()
        self.gconv_model = gconv_model
        self.loss_fcn = torch.nn.CrossEntropyLoss()

    def forward(self, graph, features, labels, train_mask):
        logits = self.gconv_model(graph, features)
        return self.loss_fcn(logits[train_mask], labels[train_mask])
