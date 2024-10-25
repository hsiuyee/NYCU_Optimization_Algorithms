#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
535526 Optimization Algorithms: HW1

This file implements the main training workflow of SVRG and SGD.

"""

import numpy as np
import argparse
import os 
import json
from datetime import datetime
import time
import torch
from torch import nn 
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler


from sgd import SGD_Vanilla
from svrg import SVRG, SVRG_Snapshot
from utils import MNIST_dataset, MNIST_logistic, MNIST_nn_one_layer, AverageCalculator, accuracy, plot_train_stats

parser = argparse.ArgumentParser(description="Train classifiers via SGD and SVRG on MNIST dataset.")
parser.add_argument('--optimizer', type=str, default="SGD", # SVRG
                    help="optimizer")
parser.add_argument('--nn_model', type=str, default="MNIST_logistic",
                    help="neural network model")
parser.add_argument('--dataset', type=str, default="MNIST", # CIFAR10
                    help="dataset")
parser.add_argument('--n_iter', type=int, default=30,
                    help="number of training iterations")
parser.add_argument('--lr', type=float, default=0.001,
                    help="learning rate")
parser.add_argument('--batch_size', type=int, default=64,
                    help="batch size")
parser.add_argument('--store_stats_interval', type=int, default=10,
                    help="how often the training statistics are stored.")

# Some macros
OUTPUT_DIR = "outputs"
BATCH_SIZE_LARGE = 256  # for validation and for the snapshots in the outer loop 
 
# Configuring the device: CPU or GPU
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print("Using device: {}".format(device))

def train_SGD_one_iter(model, optimizer, train_loader, loss_fn):
    model.train()
    loss = AverageCalculator()
    acc = AverageCalculator()
    
    for images, labels in train_loader:
        #---------- Your Code (~10 lines) ----------
        # Step 1: Flatten the images
        # Step 2: Compute loss
        # Step 3: Apply one step of SGD update by calling optimizer.step() that you defined
        
        # Step 1
        images = images.view(images.shape[0], -1).to(device)
        labels = labels.to(device)

        # Step 2
        yhat = model(images)                # compute predicted y
        loss_iter = loss_fn(yhat, labels)   # calculate loss

        # Step 3: Apply one step of SGD update by calling optimizer.step() that you defined
        optimizer.zero_grad()   # gradient = 0
        loss_iter.backward()    # back propogation
        optimizer.step()        # update parameter

        #---------- End of Your Code ----------
        # logging 
        acc_iter = accuracy(yhat, labels)
        loss.update(loss_iter.data.item())
        acc.update(acc_iter)
    
    return loss.avg, acc.avg

def train_SVRG_one_iter(model_k, model_snapshot, optimizer_inner, optimizer_snapshot, train_loader, snapshot_loader, loss_fn):
    model_k.train()
    model_snapshot.train()
    loss = AverageCalculator()
    acc = AverageCalculator()

    #---------- Your Code (~10 lines) ----------
    # calculate the mean gradient
    optimizer_snapshot.zero_grad()
    for images, labels in train_loader:
        images = images.view(images.shape[0], -1).to(device)
        labels = labels.to(device)
        yhat = model_snapshot(images)
        snapshot_loss = loss_fn(yhat, labels) / len(train_loader)
        snapshot_loss.backward()
    #---------- End of Your Code ----------

    # pass the current paramesters of optimizer_0 to optimizer_k 
    mu = optimizer_snapshot.get_param_groups()
    optimizer_inner.set_mu(mu)

    #---------- Your Code (~15 lines) ----------
    # Implment the inner loop updates
    # Step 1: Flatten the images
    # Step 2: Compute loss
    # Step 3: Apply one step of SVRG update by calling optimizer.step() that you defined  
    # Step 4: Logging the loss and accuracy in each inner loop iteration

    for images, labels in train_loader:
        # Step 1
        images = images.view(images.shape[0], -1).to(device)
        labels = labels.to(device)

        # Step 2
        yhat = model(images)
        loss_iter = loss_fn(yhat, labels)

        # Step 3
        optimizer_inner.zero_grad()
        loss_iter.backward()
        optimizer_inner.step(optimizer_snapshot.get_param_groups())

        # Step 4
        acc_iter = accuracy(yhat, labels)
        loss.update(loss_iter.data.item())
        acc.update(acc_iter)
    #---------- End of Your Code ----------    
    # update the snapshot 
    optimizer_snapshot.set_param_groups(optimizer_inner.get_param_groups())
    
    return loss.avg, acc.avg


def validate(model, val_loader, loss_fn):
    """
        Validation
    """
    model.eval()
    loss = AverageCalculator()
    acc = AverageCalculator()

    for images, labels in val_loader:
        images = images.to(device)
        images = images.view(images.shape[0], -1)
        yhat = model(images)
        labels = labels.to(device)

        # calculating loss and accuracy
        loss_iter = loss_fn(yhat, labels)
        acc_iter = accuracy(yhat, labels)
        
        # logging 
        loss.update(loss_iter.data.item())
        acc.update(acc_iter)
    
    return loss.avg, acc.avg

if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = vars(args)

    # load the MNIST dataset with the help of DataLoader in pytorch
    train_set, val_set = MNIST_dataset()
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    snapshot_loader = DataLoader(train_set, batch_size=BATCH_SIZE_LARGE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE_LARGE, shuffle=True)

    if args.nn_model == "MNIST_nn_one_layer":
        NN_model = MNIST_nn_one_layer 
    elif args.nn_model == "MNIST_logistic":
        NN_model = MNIST_logistic
    else:
        raise ValueError("Unknown model!")

    model = NN_model().to(device)
    if args.optimizer == 'SVRG':
        model_snapshot = NN_model().to(device)

    lr = args.lr  # learning rate
    n_iter = args.n_iter  # the number of training iterations
    stats_interval = args.store_stats_interval # the period of storing training statistics
    loss_fn = nn.NLLLoss()  # loss function: negative log likelihood


    # the optimizer 
    if args.optimizer == "SGD":
        optimizer = SGD_Vanilla(model.parameters(), lr=lr)
    elif args.optimizer == "SVRG":
        optimizer = SVRG(model.parameters(), lr=lr)
        optimizer_snapshot = SVRG_Snapshot(model_snapshot.parameters())
    else:
        raise ValueError("Unknown optimizer!")

    # Create a folder for storing output results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = timestamp + "_" + args.optimizer + "_" + args.nn_model
    log_dir = os.path.join(OUTPUT_DIR, model_name)
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(args_dict, f)

    # Store training stats
    train_loss_all, val_loss_all = [], []
    train_acc_all, val_acc_all = [], []

    for iteration in range(n_iter):
        t0 = time.time()

        # Training 
        if args.optimizer == "SGD":
            train_loss, train_acc = train_SGD_one_iter(model, optimizer, train_loader, loss_fn)
        elif args.optimizer == "SVRG":
            train_loss, train_acc = train_SVRG_one_iter(model, model_snapshot, optimizer, optimizer_snapshot, train_loader, snapshot_loader, loss_fn)
        else:
            raise ValueError("Unknown optimizer")
            
        # Validation 
        val_loss, val_acc = validate(model, val_loader, loss_fn)
        
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        val_loss_all.append(val_loss)  
        val_acc_all.append(val_acc)
        
        print_format = "iteration: {}, train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}, time: {:.2f} sec"
        print(print_format.format(iteration, train_loss, train_acc, val_loss, val_acc, time.time() - t0))

        # save data and plot 
        if (iteration + 1) % stats_interval == 0:
            np.savez(os.path.join(log_dir, 'train_stats.npz'), 
                train_loss=np.array(train_loss_all), train_acc=np.array(train_acc_all),
                val_loss=np.array(val_loss_all), val_acc=np.array(val_acc_all))
            plot_train_stats(train_loss_all, val_loss_all, train_acc_all, val_acc_all, log_dir, acc_low=0.9)
    
    # Training finished
    open(os.path.join(log_dir, 'done'), 'a').close()