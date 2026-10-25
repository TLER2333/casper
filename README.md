# CASPER: Identifying Perturbation Drivers of Cell State Transitions via Context-Aware Subgraph Learning

CASPER predicts driver genes underlying cellular transitions from a perturbed
single-cell profile and a reference control. It learns response-conditioned
node and edge weights over a biological prior network, combines the weighted
network readout with global cell-state context, and ranks candidate drivers.
The learned weights also provide a response-gene readout for functional analysis.

## Repository layout

```text
CASPER_main/
  casper/                         model, data loader, trainer, and metrics
  configs/                        paper configurations
  data_process/                   data split and prior-graph utilities
  start_train.py
  start_validate.py
  start_validate_double.py
  launch_train.sh
  launch_validation.sh
  launch_double_validation.sh
```

## Environment

The environment files specify Linux, Python 3.10, PyTorch 2.1.2, and CUDA 12.1.
Training defaults to one GPU; an NVIDIA GeForce RTX 4090 (24 GB) can be used.

```bash
cd CASPER_main
conda env create -f environment.yml
conda activate casper
```

The pinned `requirements.txt` can be used instead when CUDA-enabled PyTorch is
already installed.

## Expected data layout

```text
data/
  schmidt/
    train.h5ad
    test.h5ad
    gene_name.txt
    adj_matrix.csv
    dist_matrix.csv
  norman/
    single_train.h5ad
    single_test.h5ad
    double_test.h5ad
    gene_name.txt
    adj_matrix.csv
    dist_matrix.csv
  replogle_rpe1/
    train.h5ad
    val.h5ad
    test.h5ad
    gene_name.txt
    adj_matrix.csv
    dist_matrix.csv
```

Each AnnData object must contain raw counts in `layers["counts"]`,
library-size-normalized log-expression in `X`, and a categorical
`obs["perturbation"]` column. For Schmidt, Replogle RPE1, and the Norman
single-gene splits, the categorical ordering must be identical across training,
validation, and test files because it defines the classifier-to-label mapping.
Norman `double_test.h5ad` instead contains underscore-delimited gene-pair
labels, which are parsed separately by the double-perturbation evaluator.
Controls are labeled `control`.

## Data preparation

Raw datasets are not redistributed with this repository.

### Schmidt

The Schmidt raw count matrix is available from
[NCBI GEO GSE190604](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE190604),
and the associated metadata are available from
[Zenodo](https://doi.org/10.5281/zenodo.5784650). These are the data sources
used by CellNavi. The prepared files must follow the schema above and the paths
specified in `configs/schmidt.json`.

### Replogle RPE1

Download the processed Replogle RPE1 data from the
[authors' Figshare repository](https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387).

Replogle RPE1 conditions represented by fewer than 50 cells are removed, the
retained data are split 80/10/10 within condition using seed 42, and raw counts
are retained alongside library-size-normalized log-expression:

```bash
python data_process/process_Replogle_RPE1.py \
  --input /path/to/ReplogleWeissman2022_rpe1.h5ad \
  --output_dir data/replogle_rpe1 \
  --seed 42
```

### Norman

Download the expression matrix, feature and barcode files, and
`GSE133344_filtered_cell_identities.csv.gz` from
[NCBI GEO GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344).
The cell-level perturbation assignment is stored in the `guide_identity`
column. The substring before `__` identifies the guide pair. A gene paired
with `NegCtrl` is converted to a single-gene label, two gene symbols are
retained as an underscore-delimited double label (for example, `SET_KLF1`),
and non-targeting pairs are mapped to `control`.

Prepare separate single- and double-gene AnnData objects with the same ordered
5,025-gene feature vocabulary and the common AnnData schema described above.
The provided script creates the 80/20 single-gene split and the double-gene
test set from these objects. Double conditions with fewer than 200 cells are
excluded from testing; the complete control pool is retained in each split.

```bash
python data_process/process_norman_splits.py \
  --single /path/to/norman_single_hvg.h5ad \
  --double /path/to/norman_double_hvg.h5ad \
  --output_dir data/norman \
  --seed 42
```

The prior graph is constructed from the human NicheNet integrated weighted
network (`weighted_networks.rds`) released with Browaeys et al. in the
accompanying [Zenodo data record](https://zenodo.org/records/3260758).
`process_priorgraph.R` unions the directed ligand-signaling (`lr_sig`) and
gene-regulatory (`gr`) networks, retains edges whose endpoints occur in the
dataset-specific ordered gene list, and binarizes the retained weights.

Build the adjacency and shortest-path matrices with:

```bash
Rscript data_process/process_priorgraph.R \
  /path/to/weighted_networks.rds \
  data/norman/gene_name.txt \
  data/norman/adj_matrix.csv

python data_process/calculate_shortest_paths.py \
  --adjacency data/norman/adj_matrix.csv \
  --output data/norman/dist_matrix.csv
```

Run the same graph commands for Schmidt and Replogle RPE1 using their
respective ordered gene lists.

## Configurations

The paper configurations are:

- `configs/schmidt.json`: 20,000 steps, global batch size 128.
- `configs/norman.json`: 15,000 steps, global batch size 128.
- `configs/replogle_rpe1.json`: 15,000 steps, global batch size 64.

All use a six-layer, 16-head encoder with hidden dimension 256, a three-layer
graph aggregator, label smoothing 0.1, node budget 50, peak learning rate
`1e-3`, and the loss coefficients reported in the paper. The classifier has
2,048 output logits; evaluation ranks only the valid non-control categories
present in the dataset.

## Training

The command-line seed overrides the value in the
configuration and seeds Python, NumPy, PyTorch, and data-loader workers.

```bash
bash launch_train.sh configs/schmidt.json 42
bash launch_train.sh configs/norman.json 42
bash launch_train.sh configs/replogle_rpe1.json 42
```

`CUDA_VISIBLE_DEVICES` can select another GPU without changing the scripts.

## Single-perturbation evaluation

```bash
bash launch_validation.sh configs/schmidt.json 20000 42
bash launch_validation.sh configs/norman.json 15000 42
bash launch_validation.sh configs/replogle_rpe1.json 15000 42
```

The evaluator removes the control and unused classifier outputs before ranking.
It reports Top-k Accuracy, weighted F1, MRR, and nDCG@5/10 as applicable to the
candidate vocabulary. Results are written to `results/` as JSON.

## Norman double-perturbation evaluation

The Norman model is trained only on `single_train.h5ad`. Its single-gene logits
are used to rank both assigned genes in each `double_test.h5ad` profile.

```bash
bash launch_double_validation.sh configs/norman.json 15000 42
```

The evaluator reports cell-level Recall@2/5/10/50 over the 105 valid
single-gene candidates. It splits each double label at `_` and treats the two
assigned genes as an unordered target set.

## Checkpoints

Training stores checkpoints under the `model_dir` specified by each
configuration:

```text
runs/<dataset>/checkpoints/seed-<seed>/checkpoint-step-<step>.pth
```

Training logs are stored in `runs/<dataset>/logs/seed-<seed>/`.

Each checkpoint contains the model state, optimizer state, completed
optimization step, and configuration arguments. Training automatically resumes
from the checkpoint with the largest step in the selected seed directory. The
validation launchers use their seed and checkpoint-step arguments to load the
corresponding file.

Full CASPER checkpoints and accompanying reproduction artifacts will be
released upon acceptance.

## Paper-to-code map

- Paired graph encoding: `Casper._encode_graph`
- Structural scoring and aggregation: `CriticalSubgraphExtractor`
- Global information bottleneck and fusion: `Casper.predict`
- Prediction, complementary-view, VIB, and node-budget losses:
  `Casper._compute_loss`

## License

CASPER is released under the MIT License.
