#!/usr/bin/env python
import os
import sys
import json
import torch
import click
import numpy as np
from types import SimpleNamespace
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.append(os.getcwd())

from casper.model.casper import Casper as model_fn
from casper.data_provider.data_loader import ValidationLoader

def load_model(step, args):
    model = model_fn(args)
    model_path = os.path.join(args.model_dir, f"checkpoint-step-{step}.pth")
    
    print(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
        
    model.load_state_dict(new_state_dict)
    return model

def _validate_fn(model, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    loader = ValidationLoader(args)
    b_losses = defaultdict(list)
    b_metrics = defaultdict(list)
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(loader)):
            data = {k: v.to(device) for k, v in batch_data.items()}
            
            output = model(data, 0)
            
            losses = {k: v.item() for k, v in output.items() if k.endswith("loss")}
            metrics = {
                k: v.item() 
                for k, v in output.items() 
                if k.endswith("_acc") or k.startswith("token")
            }
            
            _, pred = output["pred"].topk(1, 1, True, True)
            
            y_true.extend(data["perturb"].cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().flatten().tolist())
            
            for k, v in losses.items():
                b_losses[k].append(v)
            for k, v in metrics.items():
                b_metrics[k].append(v)

            if i % 100 == 99:
                loss_str = ", ".join(
                    [f"{k}: {np.mean(v):.6f}" for k, v in b_losses.items()]
                )
                metric_str = ", ".join(
                    [f"{k}: {np.mean(v):.6f}" for k, v in b_metrics.items()]
                )
                f1 = f1_score(y_true, y_pred, average='weighted')
                print(f"{loss_str}, {metric_str} | f1 score: {f1:.6f}")
    
    final_f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Final F1 Score: {final_f1:.6f}")

@click.command()
@click.option("-c", "--checkpoint", help="Step to evaluate", required=True)
@click.option("--config", default="config.json", help="Path to config file")
def evaluate(checkpoint, config):
    with open(config, 'r') as f:
        config_dict = json.load(f)
    
    args = SimpleNamespace(**config_dict)
    args.local_rank = 0
    args.world_rank = 0

    model = load_model(checkpoint, args)
    model.eval()

    _validate_fn(model, args)

if __name__ == "__main__":
    evaluate()