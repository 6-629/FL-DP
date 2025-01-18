import argparse, json
import datetime
import os
import logging
import torch, random
import math

from server import *
from client import *
import models, datasets

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Federated Learning')

    parser.add_argument('-c', type=str, default='./utils/conf.json', dest='conf')
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)

    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])

    server = Server(conf, eval_datasets)
    clients = []

    for c in range(conf["no_models"]):
        clients.append(Client(conf, server.global_model, train_datasets, c))

    print("\n\n")
    initial_lr = conf["lr"]
    for e in range(conf["global_epochs"]):
        current_lr = initial_lr * (0.95 ** e)
        for c in clients:
            c.update_learning_rate(current_lr)
        
        server.global_model.train()
        
        candidates = random.sample(clients, conf["k"])
        weight_accumulator = {}

        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        max_norm = 5.0
        for c in candidates:
            diff = c.local_train(server.global_model)
            
            scale = 1.0 / (conf["k"] * conf["lambda"])
            for name in diff:
                diff[name] *= scale
                
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name] = weight_accumulator[name].to(diff[name].dtype)
                weight_accumulator[name].add_(diff[name])

        server.model_aggregate(weight_accumulator)
        
        server.global_model.eval()
        acc, loss = server.model_eval()

        print(f"Epoch {e}, accuracy: {acc*100:.2f}%, loss: {loss:.4f}, lr: {current_lr:.6f}")
        
        if isinstance(loss, torch.Tensor):
            if torch.isnan(loss):
                print("Training stopped due to NaN loss")
                break
        else:
            if math.isnan(loss):
                print("Training stopped due to NaN loss")
                break
