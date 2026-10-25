"""
RPE1 Data Split Script 
Inputs:
    - Raw .h5ad file containing single-cell perturbation data.
    - adata.obs must contain a 'perturbation' column.

Outputs:
    - train.h5ad, val.h5ad, test.h5ad: Split datasets with unified categorical codes.
    - gene_name.txt: List of genes retained (HVGs + target genes).
    - pert_name.txt: List of retained perturbations.

Requirements:
    - scanpy
    - numpy
    - pathlib
"""
import argparse
import scanpy as sc
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw Replogle RPE1 .h5ad")
    parser.add_argument("--output_dir", default="./data/replogle_rpe1")
    parser.add_argument("--pert_col", default="perturbation")
    parser.add_argument("--control_name", default="control")
    parser.add_argument("--min_cells", type=int, default=50,
                        help="Minimum number of cells per perturbation. Perturbations below this threshold will be discarded.")
    parser.add_argument("--n_top_genes", type=int, default=3000,
                        help="Number of highly variable genes (HVGs)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    MIN_CELLS = args.min_cells
    N_TOP_GENES = args.n_top_genes
    CONTROL_LABEL = args.control_name

    adata_path = args.input
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 1. Read data
    # ==========================================
    print(f"Reading: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    adata = adata[~adata.obs[args.pert_col].isna()].copy()

    # ==========================================
    # 2. Filter perturbations by cell count
    # ==========================================
    print(f"Filtering perturbations with < {MIN_CELLS} cells...")
    cell_counts = adata.obs[args.pert_col].value_counts()
    valid_perts = [p for p in cell_counts.index
                   if cell_counts[p] >= MIN_CELLS or p == CONTROL_LABEL]
    adata = adata[adata.obs[args.pert_col].isin(valid_perts)].copy()

    # ==========================================
    # 3. Feature selection (HVGs + Target Genes)
    # ==========================================
    print(f"Calculating HVGs (top {N_TOP_GENES})...")
    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES, flavor="seurat_v3")
    hvg_genes = set(adata.var_names[adata.var["highly_variable"]])

    # Perturbation target genes (only keep those still in var_names after filtering)
    pert_genes = set(valid_perts) & set(adata.var_names) - {CONTROL_LABEL}
    final_genes_sorted = sorted(hvg_genes | pert_genes)

    # ==========================================
    # 4. Normalization
    # ==========================================
    print("Normalizing and applying Log1p...")
    adata = adata[:, final_genes_sorted].copy()
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ==========================================
    # 5. Cell-level stratified split (80/10/10)
    # ==========================================
    print("Performing cell-level stratified split (80/10/10)...")
    np.random.seed(args.seed)

    # Crucial: Unify categorical codes before splitting to ensure consistency across splits
    all_categories = sorted(adata.obs[args.pert_col].unique().tolist())
    adata.obs[args.pert_col] = adata.obs[args.pert_col].astype("category")
    adata.obs[args.pert_col] = adata.obs[args.pert_col].cat.set_categories(all_categories)

    split_labels = np.empty(adata.n_obs, dtype=object)

    for pert, group in adata.obs.groupby(args.pert_col, observed=True):
        indices = group.index.values
        idx_arr = np.arange(len(indices))
        np.random.shuffle(idx_arr)
        n = len(indices)

        if n < 3:
            # If too few cells, put them all in train (should not trigger after filtering, kept defensively)
            split_labels[adata.obs.index.get_indexer(indices)] = "train"
        else:
            n_val = max(1, int(n * 0.1))
            n_test = max(1, int(n * 0.1))
            n_train = n - n_val - n_test

            ordered = indices[idx_arr]
            split_labels[adata.obs.index.get_indexer(ordered[:n_train])] = "train"
            split_labels[adata.obs.index.get_indexer(ordered[n_train:n_train + n_val])] = "val"
            split_labels[adata.obs.index.get_indexer(ordered[n_train + n_val:])] = "test"

    adata.obs["split"] = split_labels

    # ==========================================
    # 6. Save
    # ==========================================
    print("Saving files...")

    for split_name in ["train", "val", "test"]:
        adata_split = adata[adata.obs["split"] == split_name].copy()
        # Keep full categoricals (do not remove_unused_categories) to ensure .cat.codes consistency
        save_path = output_dir / f"{split_name}.h5ad"
        adata_split.write_h5ad(save_path)

    # gene_name.txt - needed for model gene2token mapping
    with open(output_dir / "gene_name.txt", "w") as f:
        for g in final_genes_sorted:
            f.write(g + "\n")

    # pert_name.txt - record of retained perturbations for reference
    with open(output_dir / "pert_name.txt", "w") as f:
        for p in sorted(valid_perts):
            f.write(p + "\n")

    # ==========================================
    # 7. Output configuration tips
    # ==========================================
    print(f"\n{'='*50}")
    print(f"Done! Output directory: {output_dir}")
    print(f"\nPlease set the following in config.json:")
    print(f'  effective candidates (including control): {len(all_categories)}')
    print(f'  "dataset_dir": "{output_dir}"')
    print(f'  "train_data": "{output_dir / "train.h5ad"}"')
    print(f'  "test_data": "{output_dir / "test.h5ad"}"')

if __name__ == "__main__":
    main()
