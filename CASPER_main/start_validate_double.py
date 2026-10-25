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
from casper.evaluation import category_names_from_h5ad, resolve_data_path
from casper.utils import set_seed
from start_validate import load_model


def evaluate_double(model, args):
    if not hasattr(args, "double_test_data"):
        raise ValueError("the config must define double_test_data")

    single_train_path = resolve_data_path(args.dataset_dir, args.train_data)
    candidate_names = category_names_from_h5ad(single_train_path, args.pert_col)
    valid_ids = [i for i, name in enumerate(candidate_names) if name != args.control_name]
    id_by_name = {candidate_names[i]: i for i in valid_ids}

    double_args = SimpleNamespace(**vars(args))
    double_args.test_data = args.double_test_data
    loader = ValidationLoader(double_args)
    hca_dataset = loader.dataset.dataset
    file_name = hca_dataset.file_names[0]
    adata = hca_dataset.reader.data[file_name]["adata"]
    rows = hca_dataset.perturb_rows[file_name]
    conditions = adata.obs.iloc[rows][args.pert_col].astype(str).tolist()

    delimiter = getattr(args, "double_condition_delimiter", "_")
    target_sets = []
    for condition in conditions:
        genes = condition.split(delimiter)
        if len(genes) != 2 or any(gene not in id_by_name for gene in genes):
            raise ValueError(f"invalid double condition for the single-label vocabulary: {condition}")
        target_sets.append({id_by_name[genes[0]], id_by_name[genes[1]]})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    valid_tensor = torch.tensor(valid_ids, dtype=torch.long, device=device)
    recall_sums = {2: 0.0, 5: 0.0, 10: 0.0, 50: 0.0}
    count = 0

    with torch.no_grad():
        for batch_data in tqdm(loader, desc="double-label evaluation"):
            data = {key: value.to(device) for key, value in batch_data.items()}
            masked_logits = model.predict(data)["pred"].index_select(1, valid_tensor)
            ranking = torch.argsort(masked_logits, dim=1, descending=True).cpu().numpy()

            for row in ranking:
                targets = target_sets[count]
                ranked_global_ids = [valid_ids[index] for index in row]
                for k in recall_sums:
                    recall_sums[k] += len(targets.intersection(ranked_global_ids[:k])) / 2.0
                count += 1

    if count != len(target_sets):
        raise RuntimeError(f"evaluated {count} cells but found {len(target_sets)} labels")
    return {
        "n_examples": count,
        "n_conditions": len(set(conditions)),
        "n_candidates": len(valid_ids),
        **{f"recall@{k}": recall_sums[k] / count for k in recall_sums},
    }


@click.command()
@click.option("-c", "--checkpoint", type=int, required=True, help="Norman-single checkpoint step")
@click.option("--config", default="configs/norman.json", show_default=True)
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
    metrics = evaluate_double(load_model(checkpoint, args), args)
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n")


if __name__ == "__main__":
    evaluate()
