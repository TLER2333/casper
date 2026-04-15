# CASPER: Identifying Causal Drivers of Cell State Transitions via Context-Aware Subgraph Learning
## Overview
CASPER (Context-Aware Subgraph PERturbation targets identification) is a unified framework designed to identify driver genes in cellular state transitions. By integrating biological priors with a differentiable masking mechanism, CASPER explicitly learns a "minimal sufficient subgraph" to predictively characterize state transitions, effectively filtering task-irrelevant noise from single-cell data.

Set up the Environment
To set up the development environment, we recommend using Conda. Ensure you have Python installed.

## Create a virtual environment
<pre>
conda create -n casper python=3.9
conda activate casper
pip install -r requirements.txt
</pre>

## Configuration
The training and data parameters are managed via a config.json file.

dataset_dir: The root directory containing your data files.

train_data: Filename of the training dataset (.h5ad format).

test_data: Filename of the validation/testing dataset.

dist_graph: Path to the distance matrix CSV. This represents the shortest-path distances derived from the global prior graph.

adj_graph: Path to the adjacency matrix CSV. This represents the connectivity of the global prior knowledge graph (e.g., NicheNet or other GRN/PPI).

gene_name_path: Path to a text file (.txt) containing the ordered list of gene names corresponding to the var_names in your training AnnData object.

pert_col: The column name in adata.obs indicating the perturbation label.

control_name: The label used for control cells in the pert_col column.

## start training
To initiate the training process, use the provided shell script. This script handles the distributed data parallel (DDP) setup if multiple GPUs are available.
<pre>
bash launch_train.sh
</pre>
Ensure that your run.sh points to the correct config.json and that the CUDA_VISIBLE_DEVICES are set correctly for your hardware.

## start validation
Once the training is complete, you can evaluate the model's performance on the test set and extract the identified causal subgraphs. Use the provided validation script, ensuring it points to the same config.json used during training to maintain consistency in data paths and model architecture.
<pre>
bash launch_validation.sh
</pre>
