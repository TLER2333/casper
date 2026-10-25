# ============================================================
# Function: build_adj_from_genes
#
# Description:
# This function constructs a prior graph (adjacency matrix) 
# from the NicheNet network based on a given gene list.
#
# Specifically:
# - Load the original NicheNet weighted networks (lr_sig + gr)
# - Merge them into a unified network
# - Filter edges where both genes are in the input gene list
# - Build a gene-by-gene adjacency matrix (binary: 0/1)
# - Save the adjacency matrix as a CSV file
#
# Inputs:
# - rds_path: path to NicheNet weighted_networks.rds 
#             (original prior knowledge graph from NicheNet)
# - gene_path: path to gene list file (one gene per line)
#              (genes used in model training)
# - out_path: output path for adjacency matrix CSV file
#
# Output:
# - CSV file: adjacency matrix (same gene order as input gene list)
# - Return:
#     n_nodes: number of genes
#     n_edges: number of edges in the graph
# ============================================================

build_adj_from_genes <- function(rds_path, gene_path, out_path) {
  
  library(data.table)
  library(Matrix)
  
  # ---- Load NicheNet network ----
  obj <- readRDS(rds_path)
  
  # Convert to data.table
  lr_dt <- as.data.table(obj$lr_sig)
  gr_dt <- as.data.table(obj$gr)
  
  # Merge ligand-receptor and gene regulatory networks
  net_all <- rbindlist(list(lr_dt, gr_dt), use.names = TRUE)
  
  # ---- Load gene list ----
  genes <- fread(gene_path, header = FALSE)[[1]]
  
  # Remove whitespace and duplicates
  genes <- unique(trimws(genes))
  
  # ---- Filter edges ----
  # Keep only edges where both nodes are in the gene list
  sub_edges <- net_all[
    from %in% genes & to %in% genes
  ]
  
  # ---- Build adjacency matrix ----
  # Use gene list order as fixed node order
  adj_weight <- sparseMatrix(
    i = match(sub_edges$from, genes),
    j = match(sub_edges$to, genes),
    x = sub_edges$weight,
    dims = c(length(genes), length(genes)),
    dimnames = list(genes, genes)
  )
  
  # Convert to binary adjacency (0/1)
  adj_binary <- (adj_weight != 0) * 1
  
  # ---- Save result ----
  write.csv(as.matrix(adj_binary), out_path, quote = FALSE)
  
  # ---- Return basic statistics ----
  return(list(
    n_nodes = length(genes),
    n_edges = sum(adj_binary != 0)
  ))
}


args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript process_priorgraph.R weighted_networks.rds gene_name.txt adj_matrix.csv")
}

stats <- build_adj_from_genes(
  rds_path = args[[1]],
  gene_path = args[[2]],
  out_path = args[[3]]
)
print(stats)
