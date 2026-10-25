"""Create the Norman single-gene and held-out double-gene evaluation splits."""
import argparse
from pathlib import Path

import scanpy as sc
from sklearn.model_selection import train_test_split


def split_perturbed(adata, pert_col, control_name, test_fraction, seed):
    controls = adata[adata.obs[pert_col] == control_name].copy()
    perturbed = adata[adata.obs[pert_col] != control_name].copy()
    train_indices, test_indices = [], []

    for _, group in perturbed.obs.groupby(pert_col, observed=True):
        indices = group.index.to_list()
        if len(indices) < 2:
            train_indices.extend(indices)
            continue
        train_group, test_group = train_test_split(
            indices,
            test_size=test_fraction,
            random_state=seed,
            shuffle=True,
        )
        train_indices.extend(train_group)
        test_indices.extend(test_group)

    train = sc.concat([adata[train_indices], controls], join="outer", merge="same")
    test = sc.concat([adata[test_indices], controls], join="outer", merge="same")
    categories = list(adata.obs[pert_col].cat.categories)
    for split in (train, test):
        split.obs[pert_col] = split.obs[pert_col].astype("category")
        split.obs[pert_col] = split.obs[pert_col].cat.set_categories(categories)
    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", required=True, help="Norman single-gene .h5ad")
    parser.add_argument("--double", required=True, help="Norman double-gene .h5ad")
    parser.add_argument("--output_dir", default="./data/norman")
    parser.add_argument("--pert_col", default="perturbation")
    parser.add_argument("--control_name", default="control")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_double_cells", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    single = sc.read_h5ad(args.single)
    single.obs[args.pert_col] = single.obs[args.pert_col].astype("category")
    single_train, single_test = split_perturbed(
        single, args.pert_col, args.control_name, 0.2, args.seed
    )
    single_train.write_h5ad(output_dir / "single_train.h5ad")
    single_test.write_h5ad(output_dir / "single_test.h5ad")

    double = sc.read_h5ad(args.double)
    counts = double.obs.loc[
        double.obs[args.pert_col] != args.control_name, args.pert_col
    ].value_counts()
    retained = set(counts[counts >= args.min_double_cells].index)
    keep = double.obs[args.pert_col].isin(retained | {args.control_name})
    double = double[keep].copy()
    double.obs[args.pert_col] = double.obs[args.pert_col].astype("category")
    _, double_test = split_perturbed(
        double, args.pert_col, args.control_name, 0.2, args.seed
    )
    double_test.write_h5ad(output_dir / "double_test.h5ad")


if __name__ == "__main__":
    main()
