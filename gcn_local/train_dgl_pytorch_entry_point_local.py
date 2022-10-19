import os
os.environ['DGLBACKEND'] = 'pytorch'
import dgl

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import scipy

import argparse
import logging

import time
import logging
import pickle

from sklearn.model_selection import train_test_split

from gcn_local.model.model_pytorch import GraphSAGEModel, NodeClassification
from sklearn.metrics import f1_score, accuracy_score, classification_report, balanced_accuracy_score



def save_prediction(target, pred, pred_proba, output_dir, predictions_file_name):
    
    pd.DataFrame.from_dict(
        {
            'target': target.reshape(-1, ),
            'pred_proba_class_1': pred_proba[:, 1], # estimated probabilites of class label 1
            'pred': pred.reshape(-1, ),
        }
    ).to_csv(os.path.join(output_dir, predictions_file_name), index=False)


def save_model(g, model, model_dir):
    with open(os.path.join(model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)
    with open(os.path.join(model_dir, 'graph.pkl'), 'wb') as f:
        pickle.dump(g, f)

        
def get_predictions(logits, mask, labels):
    y_pred_prob = logits[mask]
    ground_truth_labels = labels[mask]
    _, predict_labels = torch.max(y_pred_prob, dim=1)
        
    y_true = ground_truth_labels.numpy()
    y_pred = predict_labels.numpy() 
    y_pred_prob = y_pred_prob.numpy().reshape(-1, 2)
    y_pred_prob = scipy.special.softmax(y_pred_prob, axis=1)
    return y_true, y_pred, y_pred_prob


def evaluate(model, graph, features, labels, train_mask, valid_mask, test_mask=None):
    model.eval()
    with torch.no_grad():
        logits = model.gconv_model(graph, features)
        
        train_y_true, train_y_pred, train_y_prob = get_predictions(logits, train_mask, labels)
        validation_y_true, validation_y_pred, validation_y_prob = get_predictions(logits, valid_mask, labels)
        
        if test_mask is not None:
            test_y_true, test_y_pred, test_y_prob = get_predictions(logits, test_mask, labels)
            return classification_report(test_y_true, test_y_pred, zero_division=1, output_dict=True), test_y_true, test_y_pred, test_y_prob
            
        
        return accuracy_score(train_y_true, train_y_pred), f1_score(train_y_true, train_y_pred), accuracy_score(validation_y_true, validation_y_pred), f1_score(validation_y_true, validation_y_pred)


def train(model, optimizer, graph, features, labels, train_mask, val_mask, test_mask, n_epochs):
    
    duration = []
    for epoch in range(n_epochs):
        tic = time.time()
        # Set the model in the training mode.
        model.train()

        # forward
        loss = model(graph, features, labels, train_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_acc, train_f1, valid_acc, valid_f1 = evaluate(model, graph, features, labels, train_mask, val_mask)
        
        duration.append(time.time() - tic)
        logging.info(
            "Epoch {:05d} | Time(s) {:.4f} | Training Loss {:.4f} | Training F1 {:.4f} | Validation Accuracy {:.4f} | Validation F1 {:.4f}".format(
                epoch, np.mean(duration), loss.item(), train_f1, valid_acc, valid_f1)
        )

    acc_dict, y_true, y_pred_prob, y_pred = evaluate(model, graph, features, labels, train_mask, val_mask, test_mask)
    return model, acc_dict, y_true, y_pred_prob, y_pred


# def parse_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--training-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
#     parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
#     parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
#     parser.add_argument('--features-train', type=str, default="features_train.csv")
#     parser.add_argument('--features-test', type=str, default="features_test.csv")    
#     parser.add_argument('--target-column', type=str, default="binary_rating")
#     parser.add_argument('--source-destination-node-index', type=str, default="src_dst_df.csv")
#     parser.add_argument('--n-hidden', type=int, default=64)
#     parser.add_argument('--n-layers', type=int, default=2)
#     parser.add_argument('--dropout', type=float, default=0.0)
#     parser.add_argument('--weight-decay', type=float, default=5e-4)
#     parser.add_argument('--n-epochs', type=int, default=100)    
#     parser.add_argument('--lr', type=float, default=0.01)   
#     parser.add_argument('--aggregator-type', type=str, default="pool") 
#     parser.add_argument('--predictions-file-name', type=str, default="predictions.csv")

#     return parser.parse_args()



def get_logger(name):
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger.setLevel(logging.INFO)
    return logger



def entry_train(
    training_dir,
    model_dir,
    output_dir,
    features_train,
    features_test,
    target_column,
    source_destination_node_index,
    n_hidden,
    n_layers,
    dropout,
    weight_decay,
    n_epochs,
    lr,
    aggregator_type,
    predictions_file_name,

):
    logging = get_logger(__name__)
    logging.info(
        'Numpy version:{} Pytorch version:{} DGL version:{}'.format(
            np.__version__,
            torch.__version__,
            dgl.__version__,
        )
    )

    #args = parse_args()
    
    logging.info("Read train and test dataset.")
    train_validation_df = pd.read_csv(os.path.join(training_dir, features_train), header=0, index_col=0)
    test_df = pd.read_csv(os.path.join(training_dir, features_test), header=0, index_col=0)
    
    
    src_dst_df = pd.read_csv(os.path.join(training_dir, source_destination_node_index), header=0)

    logging.info("Further slice the train dataset into train and validation datasets.")
    train_df, validation_df = train_test_split(train_validation_df, test_size=0.2, random_state=42, stratify=train_validation_df[target_column].values)
    
    logging.info(f"The training data has shape: {train_df.shape}.")
    logging.info(f"The validation data has shape: {validation_df.shape}.")
    logging.info(f"The test data has shape: {test_df.shape}.")
    
    train_val_test_df = pd.concat([train_df, validation_df, test_df], axis=0)
    train_val_test_df.sort_index(inplace=True)
    
    logging.info("Generate train, validation, and test masks.")
    train_mask = torch.tensor([True if ix in set(train_df.index) else False for ix in train_val_test_df.index])
    val_mask = torch.tensor([True if ix in set(validation_df.index) else False for ix in train_val_test_df.index])
    test_mask = torch.tensor([True if ix in set(test_df.index) else False for ix in train_val_test_df.index])
    
    graph = dgl.graph((src_dst_df["src"].values.tolist(), src_dst_df["dst"].values.tolist())) # Construct graph
    # graph = dgl.to_bidirected(graph)

    features = torch.FloatTensor(train_val_test_df.drop(['node', target_column], axis=1).to_numpy()) 
    num_nodes, num_feats = features.shape[0], features.shape[1]
    logging.info(f"Number of nodes = {num_nodes}")
    logging.info(f"Number of features for each node = {num_feats}")

    labels = torch.LongTensor(train_val_test_df[target_column].values)
    n_classes = train_val_test_df[target_column].nunique()
    logging.info(f"Number of classes = {n_classes}.")

    graph.ndata['feat'] = features

    logging.info("Initializing Model")
    gconv_model = GraphSAGEModel(num_feats, n_hidden, n_classes, n_layers, F.relu, dropout, aggregator_type)
    
    # Node classification task
    model = NodeClassification(gconv_model, n_hidden, n_classes)
    logging.info("Initialized Model")

    logging.info(model)
    logging.info(model.parameters())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    logging.info("Starting Model training")
    model, metric_table, y_true, y_pred, y_pred_prob = train(model, optimizer, graph, features, labels, train_mask, val_mask, test_mask, n_epochs)
    logging.info("Finished Model training")

    logging.info("Saving model")
    save_model(graph, model, model_dir)
    
    logging.info("Saving model predictions for test data")
    save_prediction(y_true, y_pred, y_pred_prob, output_dir, predictions_file_name)
