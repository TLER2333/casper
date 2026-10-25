#!/usr/bin/env python
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import click
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.getcwd())

from casper.data_provider.data_loader import ValidationLoader
from casper.evaluation import (
    category_names_from_h5ad,
    evaluate_single_label_logits,
    resolve_data_path,
)
from casper.model.casper import Casper
from casper.utils import set_seed


def load_model(step, args):
    model = Casper(args)
    model_path = Path(args.model_dir) / f"checkpoint-step-{step}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)
    return model


def valid_candidate_ids(args):
    test_path = resolve_data_path(args.dataset_dir, args.test_data)
    names = category_names_from_h5ad(test_path, args.pert_col)
    return [i for i, name in enumerate(names) if name != args.control_name]


def evaluate_single(model, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    loader = ValidationLoader(args)

    logits, targets = [], []
    with torch.no_grad():
        for batch_data in tqdm(loader, desc="single-label evaluation"):
            data = {key: value.to(device) for key, value in batch_data.items()}
            output = model.predict(data)
            logits.append(output["pred"].detach().cpu().numpy())
            targets.append(data["perturb"].detach().cpu().numpy())

    logits = np.concatenate(logits, axis=0)
    targets = np.concatenate(targets, axis=0)
    return evaluate_single_label_logits(logits, targets, valid_candidate_ids(args))


@click.command()
@click.option("-c", "--checkpoint", type=int, required=True, help="Checkpoint step")
@click.option("--config", default="configs/schmidt.json", show_default=True)
@click.option("--seed", type=int, default=None, help="Override the config seed")
@click.option("--output", type=click.Path(dir_okay=False), default=None)
def evaluate(checkpoint, config, seed, output):
    with open(config, "r") as handle:
        config_dict = json.load(handle)
    if seed is not None:
        config_dict["seed"] = seed
    config_dict.setdefault("seed", 42)
    config_dict.setdefault("deterministic", False)
    args = SimpleNamespace(**config_dict)
    args.local_rank = 0
    args.world_rank = 0
    args.model_dir = str(Path(args.model_dir) / f"seed-{args.seed}")

    set_seed(args.seed, deterministic=args.deterministic)
    metrics = evaluate_single(load_model(checkpoint, args), args)
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n")


if __name__ == "__main__":
    evaluate()
